from functools import partial
from typing import Tuple, List, Dict, Any, Union, Optional
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import scipy.special
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

from scipy.spatial.distance import cosine
from scipy.special import rel_entr
import random
import matplotlib as mpl
#import colorcet  # noqa
from tqdm import tqdm
from langdetect import detect, DetectorFactory
import plotly.graph_objects as go
from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names
from functools import lru_cache
#from quant_utils.ptq_configs.bitnet_ptq import apply_bitnet_ptq



# ---------------------------
# Lang Detect - NOT IMPLEMENTED
# ---------------------------
DetectorFactory.seed = 0  # for reproducibility

# Build token -> language map
def build_token_language_map(tokenizer:Any) -> Dict:
    token_lang_map = {}
    for token_id in tqdm(range(tokenizer.vocab_size)):
        token_str = tokenizer.decode([token_id])
        try:
            lang = detect(token_str)
        except:
            lang = "unknown"
        token_lang_map[token_id] = lang
    
    return token_lang_map


def compute_language_coverage(topk_indices:Any, token_lang_map:Dict, target_languages:Any|None=None) -> List:
    lang_coverage_per_layer = []
    for i in range(len(topk_indices)):
        lang_counts = {}
        for j in range(topk_indices.shape[1]):
            for k in range(topk_indices.shape[2]):
                token_id = topk_indices[i, j, k]
                lang = token_lang_map.get(token_id, "unknown")
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        # Compute coverage
        total = sum(lang_counts.values())
        if target_languages is None:
            target_languages = list(lang_counts.keys())
        lang_coverage = {lang: lang_counts.get(lang, 0) / total for lang in target_languages}
        lang_coverage_per_layer.append(lang_coverage)
    
    return lang_coverage_per_layer



# ---------------------------
# Configs, load model & tokenizer
# ---------------------------
def _set_deterministic_backend(seed:int=42) -> None:
    """ 
    Forces PyTorch to use only deterministic operations (e.g., disables non-deterministic GPU kernels).
    This is crucial for reproducibility: given the same inputs and model state, to get the same outputs every time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clear_cuda_cache() -> None:
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()


def save_metrics_to_json(metrics_list:List[Dict], save_path:str) -> None:
    def convert_ndarray(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
            return o.item()
        elif isinstance(o, dict):
            return {k: convert_ndarray(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_ndarray(i) for i in o]
        return o

    serializable_metrics = convert_ndarray(metrics_list)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


@lru_cache(maxsize=2)
def _load_model_tokenizer(model_id:str, tok_id:str, quant_config:str|None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)

    if quant_config:
        if '4' in quant_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=(quant_config == 'ptsq4bit'),
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16 
            )
        elif '8' in quant_config:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16 
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='auto',
            return_dict=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            return_dict=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True
        )

    return model, tok



# ---------------------------
# Safecasting
# ---------------------------
"""def safe_cast_logits(tensor:torch.Tensor) -> torch.Tensor:
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8]:
        tensor = tensor.to(torch.float32)
    return torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)"""
def safe_cast_logits(tensor:torch.Tensor) -> torch.Tensor:
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8]:
        tensor = tensor.to(torch.float32)

    tensor = torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)

    # Clamp extreme values just in case
    tensor = torch.clamp(tensor, min=-1e5, max=1e5)
    return tensor


"""def numpy_safe_cast(x:Any) -> Any:
    x = x.astype(np.float32)
    return np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)"""
def numpy_safe_cast(x:Any) -> Any:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)
    return np.clip(x, -1e5, 1e5)


# ---------------------------
# Make lens hooks
# ---------------------------
def topk_make_lens_hooks(model:Any, layer_names:Any, verbose=False) -> List:
    hook_handles = []
    
    print(f"[Debug] Layer names being passed: {layer_names}")

    for layer_name in layer_names:
        try:
            print(f"[Debug] Trying to access layer: {layer_name}")
            layer = dict(model.named_modules())[layer_name]  # Get the layer by name
            print(f"[Debug] Successfully found layer: {layer_name}")

            # Register the hook for the layer
            handle = layer.register_forward_hook(my_hook)
            hook_handles.append(handle)
        
        except KeyError:
            print(f"[Error] Layer {layer_name} not found in model.")  # If the layer is not found
        except Exception as e:
            print(f"[Error] Failed to register hook for {layer_name}: {str(e)}")

    if verbose:
        print(f"[Debug] Hook handles: {hook_handles}")

    return hook_handles


def my_hook(module:Any, input:Any, output:Any) -> Any:
    print(f"[Hook] Layer {module} received input {input} and output {output}")
    return output 


def make_layer_names_topk(model:Any) -> List:
    # Generate layer names for LLaMA
    layer_names = []

    # Access the layers from the model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # This model is structured with a 'model' submodule that contains 'layers'
        for i in range(len(model.model.layers)):
            layer_names.append(f"model.layers.{i}")
        layer_names.append("model.embed_tokens")  # Add the embedding layer

    elif hasattr(model, 'layers'):
        # For models without the 'model' submodule
        for i in range(len(model.layers)):
            layer_names.append(f"layers.{i}")

    else:
        print("[Error] Cannot find layers in the model.")
    
    return layer_names



# ---------------------------
# Tokenize
# ---------------------------
def text_to_input_ids(
        tokenizer:Any,
        text:Union[str, List[str]],
        model:Optional[torch.nn.Module]=None,
        add_special_tokens:bool=True,
        pad_to_max_length=False
) -> torch.Tensor:
    """
    Tokenize the inputs, respecting padding behavior.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # EOS

    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Padding to the longest sequence in the batch or to max length
    tokens = tokenizer(
        texts,
        return_tensors='pt',
        padding='longest' if not pad_to_max_length else True,  # Padding only to longest sequence
        truncation=True,
        add_special_tokens=add_special_tokens,
    )['input_ids']

    if model is not None:
        device = getattr(model, 'device', None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokens = tokens.to(device)

    return tokens  # shape: [batch_size, seq_len]



# ---------------------------
# Collect & preprocess logits
# ---------------------------
"""def collect_logits(model:Any, input_ids:Any, layer_names:Any, decoder_layer_names:Optional[Any|None]) -> Tuple:
    model._last_resid = None
    
    # Handle single vs batch input
    if input_ids.ndim == 2:  # Single prompt
        batch_size = 1
    else:  # Multiple prompts (batch)
        batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        out = model(input_ids)
    del out
    model._last_resid = None
    
    # gather logits for each layer (whether for single or batch)
    try:
        layer_logits = np.concatenate(
            [model._layer_logits.get(name, np.zeros((batch_size, model.config.hidden_size))) for name in layer_names],
            axis=0,
        )
    except KeyError as e:
        print(f"[Error] Missing layer logits for {e}")
        layer_logits = np.zeros((batch_size, len(layer_names), model.config.hidden_size))

    return layer_logits, layer_names"""
def collect_logits(model:Any, input_ids:Any, layer_names:Any, decoder_layer_names:Any|None=None) -> Tuple[Any,Any]:
    model._last_resid = None

    with torch.no_grad():
        _ = model(input_ids, output_hidden_states=True)

    model._last_resid = None

    logits_by_layer = []
    valid_layer_names = []
    for name in layer_names:
        layer_output = model._layer_logits.get(name)
        if layer_output is None:
            raise ValueError(f"Missing logits for layer: {name}")

        # Convert to numpy if tensor
        if isinstance(layer_output, torch.Tensor):
            layer_output = layer_output.detach().cpu().numpy()

        # Validate shape: expecting 4D (L, B, T, V) or 3D (B=1, T, V) after removing batch
        if layer_output.ndim == 4:
            # Shape good for typical multi-batch, multi-layer outputs, proceed
            pass
        elif layer_output.ndim == 3 and layer_output.shape[0] == 1:
            layer_output = layer_output[0]  # remove batch dim
        else:
            print(f"Warning: Skipping layer '{name}' due to unexpected shape {layer_output.shape}")
            continue  # skip this layer

        logits_by_layer.append(layer_output)
        valid_layer_names.append(name)

    if len(logits_by_layer) == 0:
        raise RuntimeError("No valid logits found for any layers!")

    layer_logits = np.stack(logits_by_layer, axis=0)  # (num_layers, seq_len, vocab_size)

    return layer_logits, valid_layer_names



def collect_batch_logits(model, input_ids, layer_names, outputs) -> Tuple:
    collected_logits = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Get hidden states only
        collected_logits.append(output.detach().cpu().numpy())

    handles = []
    for name in layer_names:
        layer = dict([*model.named_modules()])[name]
        handles.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_ids)

        for h in handles:
            h.remove()

    # Stack into shape [num_layers, batch, seq_len, hidden_size]
    logits = np.stack(collected_logits, axis=0)

    return logits, layer_names


def postprocess_logits_topk(
        layer_logits:Any,
        normalize_probs=False,
        top_n:int=5,
        return_scores:bool=True,
        safe_cast:bool=False
) -> Tuple[Any, Any, Any]:

    if safe_cast:
        layer_logits = layer_logits.astype(np.float32)

    # Replace NaNs and infs with appropriate values
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    # Normalize the probabilities if needed
    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    # Get the index of the maximum logit (predicted token)
    layer_preds = layer_logits.argmax(axis=-1)

    # Compute the mean of the top-N probabilities for each token
    top_n_scores = np.mean(
        np.sort(layer_probs, axis=-1)[:, -top_n:], axis=-1
    )

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs
    


# ---------------------------
# Stability, corretness, accuracy scores
# ---------------------------
def calculate_avg_stability(stability_top1:Any, stability_topk:Any) -> Tuple[Any, Any]:
    """
    Calculate the average stability score based on top-1 and top-k metrics.
    
    Args:
        stability_top1 (np.ndarray): Stability metric for top-1 predictions, shape = [tokens]
        stability_topk (np.ndarray): Stability metric for top-k predictions, shape = [tokens]
    
    Returns:
        tuple: average_stability_top1, average_stability_topk
    """
    avg_stability_top1 = np.mean(stability_top1[stability_top1 != -1])  # Average stability for top-1 predictions
    avg_stability_topk = np.mean(stability_topk[stability_topk != -1])  # Average stability for top-k predictions

    return avg_stability_top1, avg_stability_topk


def compute_safe_stability_metrics(layer_preds, topk_indices, target_ids) -> Tuple[Any,Any]:
    """
    Compute the stability metrics excluding layer 0 (embedding/projection layer).
    """
    # Slice off layer 0
    layer_preds = layer_preds[1:]  # [layers-1, tokens]
    topk_indices = topk_indices[1:]  # [layers-1, tokens, topk]
    
    # Top-1 Stability
    match_top1 = layer_preds == target_ids
    has_match_top1 = np.any(match_top1, axis=0)
    stability_top1 = np.where(has_match_top1, np.argmax(match_top1, axis=0) + 1, -1)

    # Top-k Stability
    match_topk = np.any(topk_indices == target_ids[None, :, None], axis=-1)
    has_match_topk = np.any(match_topk, axis=0)
    stability_topk = np.where(has_match_topk, np.argmax(match_topk, axis=0) + 1, -1)

    return stability_top1, stability_topk


def compute_stability_metrics(layer_preds:Any, topk_indices:Any, target_ids:Any) -> Tuple[Any,Any]:
    """
    Compute the stability metrics: how quickly the model predicts the correct token 
    in top-1 and top-k predictions relative to the depth (layers).

    Args:
        layer_preds (np.ndarray): Predicted tokens for each layer, shape = [layers, tokens]
        topk_indices (np.ndarray): Indices of the top-k predicted tokens for each layer, shape = [layers, tokens, topk]
        target_ids (np.ndarray): Ground truth token ids, shape = [tokens]

    Returns:
        tuple: stability_top1, stability_topk
    """
    
    # 1. Stability in terms of top-1 prediction: Find the first layer where the model's top-1 prediction is correct
    stability_top1 = np.argmax(layer_preds == target_ids, axis=0)  # First layer where correct token is top-1
    stability_top1 = np.where(stability_top1 == 0, -1, stability_top1)  # If no layer is correct, return -1
    
    # 2. Stability in terms of top-k prediction: Find the first layer where the correct token is in the top-k
    stability_topk = np.argmax(np.any(topk_indices == target_ids[None, :, None], axis=-1), axis=0)  # First layer where correct token is in top-k
    stability_topk = np.where(stability_topk == 0, -1, stability_topk)  # If no layer is correct, return -1
    
    return stability_top1, stability_topk


def compute_correctness_metrics(layer_preds:Any, target_ids:Any, topk_indices:Any|None=None) -> Tuple[Any,Any]:
    correct_1 = (layer_preds[-1] == target_ids).astype(int)

    if topk_indices is not None:
        # target_ids: [tokens], reshape to [1, tokens, 1] to broadcast against [layers, tokens, topk]
        target_ids_broadcasted = target_ids[None, :, None]
        correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)
    else:
        correct_topk = None

    return correct_1, correct_topk



# ---------------------------
# Metric & clipping helpers
# ---------------------------
def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark

def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)

def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)

def kl_div(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)

def compute_entropy(probs):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def maybe_batchify(p):
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p

def min_max_scale(arr, new_min, new_max):
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

def compute_kl_divergence(logits1, logits2):
    probs1 = scipy.special.softmax(logits1, axis=-1)
    probs2 = scipy.special.softmax(logits2, axis=-1)
    return np.sum(kl_div(probs1, probs2), axis=-1)

def compute_cosine_similarity_across_layers(layer_probs):
    cosine_sims = []
    for i in range(len(layer_probs)-1):
        sims_per_token = []
        for j in range(layer_probs.shape[1]):  # token positions
            p1 = layer_probs[i, j]
            p2 = layer_probs[i+1, j]
            sim = 1 - cosine(p1, p2)
            sims_per_token.append(sim)
        cosine_sims.append(np.mean(sims_per_token))
    return cosine_sims

def compute_kl_vs_previous_layer(layer_probs):
    kl_per_layer = []
    for i in range(len(layer_probs)-1):
        kl_tokens = []
        for j in range(layer_probs.shape[1]):  # token positions
            p1 = np.clip(layer_probs[i, j], 1e-10, 1.0)
            p2 = np.clip(layer_probs[i+1, j], 1e-10, 1.0)
            kl = np.sum(rel_entr(p1, p2))
            kl_tokens.append(kl)
        kl_per_layer.append(np.mean(kl_tokens))
    return kl_per_layer

def compute_token_variety(topk_indices):
    unique_tokens_per_layer = []
    for i in range(len(topk_indices)):
        unique_tokens = np.unique(topk_indices[i])
        unique_tokens_per_layer.append(len(unique_tokens))
    return unique_tokens_per_layer

def expand_metric_for_heatmap(metric_values, num_tokens):
    return np.repeat(np.expand_dims(np.array(metric_values), axis=1), num_tokens, axis=1)

def expand_transition_metric_for_heatmap(metric_values, num_layers, num_tokens):
    # Pad with one value (e.g. repeat first or last value) to make it num_layers long
    padded_metric_values = np.concatenate([[metric_values[0]], metric_values])
    return np.repeat(np.expand_dims(np.array(padded_metric_values), axis=1), num_tokens, axis=1)



# ---------------------------
# Topk batch analysis & logging
# ---------------------------
def collect_logit_lens_metrics_batch(
    model:Any,
    tokenizer:Any,
    prompts:List[str],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    prompt_type:str='text',
    max_prompts:int=50,
) -> List:
    assert isinstance(prompts, list), "prompts should be a list of strings"
    prompts = prompts[:max_prompts]

    results = []

    for idx, prompt in enumerate(prompts):
        input_ids_tensor = text_to_input_ids(tokenizer, prompt, model)
        input_ids_list = input_ids_tensor[0].tolist()

        layer_names = make_layer_names_topk(model)

        hook_handles = topk_make_lens_hooks(model, layer_names=layer_names)
        if hook_handles is None:
            print(f"[Error] No hooks were registered for prompt {idx}. Skipping.")
            continue

        try:
            layer_logits, _ = collect_batch_logits(model, input_ids_tensor, layer_names, [])

            # Handle shape: [layers, batch, seq_len, hidden] → [layers, seq_len, hidden]
            if isinstance(layer_logits, list):
                layer_logits = np.stack(layer_logits, axis=0)
            if layer_logits.ndim == 4 and layer_logits.shape[1] == 1:
                layer_logits = layer_logits[:, 0, :, :]
            elif layer_logits.ndim != 3:
                raise ValueError(f"Expected layer_logits to be 3D but got shape {layer_logits.shape}")

            # Project to vocab if it's still in hidden state space
            if layer_logits.shape[-1] == model.config.hidden_size:
                hidden_states = torch.tensor(layer_logits, dtype=torch.float32).to(model.device)
                with torch.no_grad():
                    logits = model.lm_head(hidden_states)
                layer_logits = logits.cpu().numpy()
                # Clean logits immediately
                layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            # Slice to match the token prediction window
            layer_logits = layer_logits[:, start_ix + 1:end_ix + 1, :]

            # Top-k prediction postprocessing
            layer_preds, layer_probs, _ = postprocess_logits_topk(layer_logits, top_n=topk)
            topk_indices = np.argsort(layer_probs, axis=-1)[..., -topk:][..., ::-1]

            # Ground truth target token IDs
            target_ids = input_ids_tensor[0, start_ix + 1:end_ix + 1].cpu().numpy()

            # Metrics: entropy, correctness
            entropy = compute_entropy(layer_probs)
            prob_correct = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1).squeeze(-1)
            logit_correct = np.take_along_axis(layer_logits, layer_preds[..., None], axis=-1).squeeze(-1)

            # Broadcast target IDs for top-k correctness
            target_ids_broadcasted = target_ids[None, :, None]
            correct_1 = (layer_preds == target_ids[None, :]).astype(int)
            correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)

            # Stability metrics
            stability_top1, stability_topk = compute_stability_metrics(layer_preds, topk_indices, target_ids)
            safe_stability_top1, safe_stability_topk = compute_safe_stability_metrics(layer_preds, topk_indices, target_ids)

            # Aggregated stats
            correct_1_std = np.std(correct_1, axis=1).tolist()
            correct_topk_std = np.std(correct_topk, axis=1).tolist()
            vocab_size = tokenizer.vocab_size
            norm_entropy = (entropy / np.log(vocab_size)).tolist()

            # KL divergence between layers
            layer_kl_divergences = [
                compute_kl_divergence(layer_logits[i], layer_logits[i + 1])
                for i in range(len(layer_logits) - 1)
            ]

            # Store results
            metrics = {
                "prompt": input_ids_tensor.tolist(),
                "decoded_prompt_str": tokenizer.decode(input_ids_list),
                "tokens": tokenizer.convert_ids_to_tokens(input_ids_list),
                "prompt_type": prompt_type,
                "target_ids": target_ids.tolist(),
                "target_tokens": tokenizer.convert_ids_to_tokens(target_ids.tolist()),
                "layer_names": layer_names,
                "correct_1": correct_1.mean(axis=1).tolist(),
                "correct_topk": correct_topk.mean(axis=1).tolist(),
                "correct_1_std": correct_1_std,
                "correct_topk_std": correct_topk_std,
                "correct_1_by_position": correct_1.T.tolist(),
                "correct_topk_by_position": correct_topk.T.tolist(),
                "entropy": entropy.mean(axis=1).tolist(),
                "normalized_entropy": norm_entropy,
                "logit_mean": logit_correct.mean(axis=1).tolist(),
                "prob_mean": prob_correct.mean(axis=1).tolist(),
                "stability_top1": stability_top1.tolist(),
                "stability_topk": stability_topk.tolist(),
                "safe_stability_top1": safe_stability_top1.tolist(),
                "safe_stability_topk": safe_stability_topk.tolist(),
                "layer_kl_divergences": layer_kl_divergences,
            }

            results.append(metrics)

        finally:
            if hook_handles:
                for handle in hook_handles:
                    handle.remove()
            for var in ['input_ids_tensor', 'layer_logits', 'layer_preds', 'layer_probs']:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()

    return results



# ---------------------------
# Logit Lens heatmap
# ---------------------------
def _topk_logit_lens_fig(
    layer_logits:Any,
    layer_preds:Any,
    layer_probs:Any,
    topk_scores:Any,
    topk_indices:Any,
    tokenizer:Any,
    input_ids:Any,
    start_ix:int,
    layer_names:Any,
    top_k=5,
    topk_mean:bool=True,
    normalize=True,
    metric_type:str|None=None,
    map_color='Cividis',
    value_matrix=None,
    rank_matrix_raw:Optional[Any|None]=None,
    pred_ranks:Optional[Any|None]=None,    
    title:str|None=None,
    block_step:int=1,
    token_font_size:int=12,
    label_font_size:int=20,
) -> go.Figure:
    
    num_layers, num_tokens, vocab_size = layer_logits.shape
    end_ix = start_ix + num_tokens

    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    # Top-axis label logic
    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < num_tokens:
        next_token_text.append("")

    # Predicted token text for each layer/token
    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)

    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    # Mean top-k probability as default value_matrix if not provided
    if value_matrix is None:
        value_matrix = topk_scores.mean(axis=-1)

    # Layer filtering using block_step
    keep_idxs = [0] + list(range(1, num_layers - 1, block_step)) + [num_layers - 1]
    value_matrix = value_matrix[keep_idxs]
    pred_token_text = pred_token_text[keep_idxs]
    layer_preds = layer_preds[keep_idxs]
    topk_scores = topk_scores[keep_idxs]
    topk_tokens = topk_tokens[keep_idxs]
    layer_names = [layer_names[i] for i in keep_idxs]

    if metric_type == 'ranks':
        if pred_ranks is None:
            raise ValueError("pred_ranks must be provided for metric_type='ranks'")
        pred_ranks = pred_ranks[keep_idxs]

    # ── Hover-text and normalization logic ─────────────────────
    hovertext = []

    for i in range(len(layer_names)):
        row = []
        for j in range(num_tokens):
            if metric_type == "ranks":
                raw_val = pred_ranks[i, j]
            else:
                raw_val = value_matrix[i, j]

            # Hover display logic
            try:
                if metric_type == "entropy":
                    hover_val = f"<b>Entropy:</b> {raw_val:.3f}<br>"
                elif metric_type == "logits":
                    hover_val = f"<b>Logit:</b> {raw_val:.3f}<br>"
                elif metric_type == "kl":
                    hover_val = f"<b>KL Divergence:</b> {raw_val:.3f}<br>"
                elif metric_type == "cos_sims":
                    hover_val = f"<b>Cosine Similarity:</b> {raw_val:.3f}<br>"
                elif metric_type == "lw_kl":
                    hover_val = f"<b>LW KL Div:</b> {raw_val:.3f}<br>"
                elif metric_type == "tok_var":
                    hover_val = f"<b>Token Variety:</b> {raw_val:.3f}<br>"
                elif metric_type == "ranks":
                    hover_val = f"<b>Rank:</b> {raw_val}<br>"
                else:
                    hover_val = f"<b>Mean Top-{top_k} Prob:</b> {raw_val:.3f}<br>"
            except:
                hover_val = f"<b>{metric_type}:</b> N/A<br>"

            # include predicted token and top-k predictions
            hover = hover_val
            hover += f"<b>Pred:</b> {pred_token_text[i, j]}<br><b>Top-{top_k}:</b><br>"
            for k in range(top_k):
                hover += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}: {topk_scores[i, j, k]:.3f}<br>"

            row.append(hover)
        hovertext.append(row)

    # ── Normalization of ranks for coloring ────────────────────
    if normalize:
        if metric_type == 'ranks':
            # Normalize ranks for coloring (log scale)
            vmax = 2000  # Max value for the rank
            norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)  # LogNorm for better color scaling

            # Apply log scale and normalization to the value matrix for proper coloring
            value_matrix = np.log10(pred_ranks)  # Use log scale for ranks
            value_matrix = min_max_scale(value_matrix, 0, 1)  # Scale to [0, 1] for proper coloring
        
        elif metric_type == 'kl' or metric_type == 'lw_kl' or metric_type == 'entropy':
            value_matrix = min_max_scale(value_matrix, 0, 10)

        else:
            value_matrix = min_max_scale(value_matrix, 0, 1)

    # Decode predictions from selected layers
    pred_tokens_str = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)

    # Prepare true next tokens
    true_next_token_text = next_token_text[:num_tokens]
    if len(true_next_token_text) < num_tokens:
        true_next_token_text += [""] * (num_tokens - len(true_next_token_text))

    # Ground truth matrix for comparison: shape [layers x tokens]
    correct_tokens_matrix = np.tile(true_next_token_text, (pred_tokens_str.shape[0], 1))

    # Check which predictions match true next tokens
    is_correct = (pred_tokens_str == correct_tokens_matrix)

    # Decode input tokens (used to check for embedding projection)
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0][start_ix:end_ix]]
    input_tokens_matrix = np.tile(input_tokens_str, (pred_tokens_str.shape[0], 1))

    # Suppress early-layer predictions that just repeat the input token (likely embedding projection)
    not_input_projection = (pred_tokens_str != input_tokens_matrix)

    # A mask: for each (layer, token)
    echo_mask = (pred_tokens_str == input_tokens_matrix)

    for j in range(echo_mask.shape[1]):  # loop over token positions
        # If any previous layer (above current one) predicted something else, keep it
        for i in range(1, echo_mask.shape[0]):
            if not echo_mask[i - 1, j]:
                # a prior non-echo happened, so from here on allow it even if it's an echo again
                echo_mask[i:, j] = False
                break

    # Filter: suppress only those that are still pure echoes
    is_correct = is_correct & ~echo_mask

    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    is_correct = is_correct[::-1]
    
    fig = go.Figure()

    # Use rank values for text display if plotting ranks, else predicted tokens
    if metric_type == 'ranks':
        cell_text = pred_ranks[::-1].astype(str)  # Match flipped value_matrix
    else:
        cell_text = pred_token_text

    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(num_tokens)),
        y=list(range(len(layer_names))),
        text=cell_text,
        #text=pred_token_text,
        texttemplate="%{text}",
        textfont=dict(size=token_font_size),
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        zmin=np.min(value_matrix),
        zmax=np.max(value_matrix),
        colorbar=dict(
            title="log₁₀(rank)" if metric_type == 'ranks' else None
        ),
    ))

    correct_y, correct_x = np.where(is_correct == 1)
    for y, x in zip(correct_y, correct_x):
        fig.add_shape(
            type='rect',
            x0=x - 0.5, x1=x + 0.5,
            y0=y - 0.5, y1=y + 0.5,
            line=dict(color='black', width=2),
            layer='above'
        )

    fig.add_trace(go.Scatter(
        x=list(range(num_tokens)),
        y=[None] * num_tokens,
        xaxis='x2',
        mode='markers',
        marker=dict(opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        font=dict(family="DejaVu Sans", size=label_font_size), 
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(num_tokens)),
            ticktext=input_tokens_str,
            title='Input Token',
            side='bottom',
            anchor='y',
            domain=[0.0, 1.0]
        ),
        xaxis2=dict(
            tickmode='array',
            tickvals=list(range(num_tokens)),
            ticktext=next_token_text,
            overlaying='x',
            side='top',
            anchor='free',
            position=1.0,
            showline=True,
            ticks='outside'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(layer_names))),
            ticktext=layer_names[::-1],
            title='Layer',
            autorange='reversed',
        ),
        width=max(1200, 100 * num_tokens),
        height=max(600, 35 * len(layer_names)),
        margin=dict(l=20, r=10, t=40, b=10),
    )

    return fig



# ---------------------------
# The topk Logit Lens plotter
# ---------------------------
def plot_topk_logit_lens(
    model_path:str,
    tokenizer_path:str,
    inputs:Union[str, List[str], None],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    topk_mean:bool=True,
    lang_detect:str|None=None,
    probs:bool=False,
    entropy:bool=False,
    kl:bool=False,
    cosine_sims:bool=False,
    kl_layerwise:bool=False,
    token_variety:bool=False,
    ranks:bool=False,
    block_step:int=1,
    token_font_size:int=12,
    label_font_size:int=20,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:List=['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False,
    pad_to_max_length:bool=False,
    model_precision:Optional[str|None]=None,
    use_deterministic_backend:bool=False,
    json_log_path:str|None=None,
    safe_cast:bool=False # true -> np.float32 
) -> go.Figure:
    """ Plot topk Logit Lens """

    # ---- load model, tokenizer
    model, tokenizer = _load_model_tokenizer(model_path, tokenizer_path, model_precision)

    # Only call .to() if it's not a quantized model
    is_quantized = hasattr(model, 'is_loaded_in_4bit') or hasattr(model, 'is_loaded_in_8bit')
    if model_precision and not is_quantized:
        model = model.to(model_precision)

    # ---- suppress errors, set deterministic backend if needed
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    if use_deterministic_backend:
        _set_deterministic_backend()
    
    rank_matrix_raw = None
    pred_ranks = None
    metric_type = None

    # ---- if not logging, compute, plot logit lens
    if json_log_path is None:
        if isinstance(inputs, str):
            inputs = [inputs]
        elif inputs is None:
            inputs = ["What is y if y=2*2-4+(3*2)"]  # Default prompt

        layer_names = make_layer_names(
            model,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        # ---- make lens hooks
        make_lens_hooks(model, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names, decoder_layer_names=decoder_layer_names, verbose=verbose)
        
        # ---- tokenize
        input_ids = text_to_input_ids(tokenizer, inputs, model, pad_to_max_length=pad_to_max_length)

        # ---- collect, preprocess logits
        layer_logits, layer_names = collect_logits(model, input_ids, layer_names, decoder_layer_names)

        if safe_cast:
            layer_logits = safe_cast_logits(torch.tensor(layer_logits)).numpy()

        layer_preds, layer_probs, _ = postprocess_logits_topk(layer_logits, top_n=topk, safe_cast=safe_cast)

        # ---- clean probs
        layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

        # compute top-k indices and scores from cleaned probs
        topk_indices = np.argsort(layer_probs, axis=-1)[..., -topk:][..., ::-1]
        topk_scores = np.take_along_axis(layer_probs, topk_indices, axis=-1)

        # ---- cases
        if lang_detect:
            map_color = 'RdBu_r'
            """pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
            pred_probs_full = np.zeros_like(layer_probs)
            np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
            
            token_lang_map = build_token_language_map(tokenizer=tokenizer)
            lang_coverage_per_layer = compute_language_coverage(topk_indices=topk_indices, token_lang_map=token_lang_map, target_languages=None)
            coverage = [layer_cov.get(lang_detect, 0.0) for layer_cov in lang_coverage_per_layer]

            value_matrix = expand_transition_metric_for_heatmap(coverage, len(layer_probs), layer_probs.shape[1])

            metric_type = 'lang'
            title = f"{lang_detect} Language Coverage ({'mean topk' if topk_mean else 'top-1'})"
            """
            raise NotImplementedError("Not implemented yet!")
        
        # Entropy (mean over top-k only)
        elif entropy:
            map_color = 'RdBu_r'
            if topk_mean:
                clipped_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)
                log_probs = np.log(np.clip(clipped_probs, 1e-10, 1.0))
                value_matrix = -np.sum(clipped_probs * log_probs, axis=-1)
            else:
                value_matrix = compute_entropy(layer_probs)
            metric_type = 'entropy'
            title = f"Entropy ({'mean topk' if topk_mean else 'full dist'})"

        # Probabilities
        elif probs:
            map_color = 'Blues'
            if topk_mean:
                value_matrix = topk_scores.mean(axis=-1)
            else:
                value_matrix = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1).squeeze(-1)
            metric_type = 'probs'
            title = f"Probabilities ({'mean topk' if topk_mean else 'top-1'})"

        # KL-Divergence block
        elif kl: 
            map_color = 'Cividis'

            if topk_mean:
                clipped_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)  # (L, T, K)
                log_probs = np.log(np.clip(clipped_probs, 1e-10, 1.0))
                q_probs = np.exp(log_probs)  # Just to be safe: ensure proper probs (though they already are)

                # Scatter q_probs into full vocab-size tensor
                q_full_probs = np.zeros_like(layer_probs)
                for k in range(topk_indices.shape[-1]):
                    np.put_along_axis(
                        q_full_probs,
                        topk_indices[:, :, k:k+1],
                        q_probs[:, :, k:k+1],
                        axis=-1
                    )

                # Compute KL divergence between full probs and top-k projected
                value_matrix = kl_div(layer_probs, q_full_probs)
            
            else:
                # Just compare full predicted distribution to one-hot of top-1 pred
                pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
                pred_probs_full = np.zeros_like(layer_probs)
                np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
                value_matrix = kl_div(layer_probs, pred_probs_full)

            metric_type = 'kl'
            title = f"KL Divergence ({'mean topk' if topk_mean else 'top-1'})"

        # Cosine similarities block
        elif cosine_sims:
            map_color = 'Reds'

            pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
            pred_probs_full = np.zeros_like(layer_probs)
            np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
            
            cosine_sims_values = compute_cosine_similarity_across_layers(layer_probs)
            value_matrix = expand_transition_metric_for_heatmap(cosine_sims_values, len(layer_probs), layer_probs.shape[1])

            metric_type = 'cos_sims'
            title = f"Cosine Similarity ({'mean topk' if topk_mean else 'top-1'})"

        # Layer-wise KL-Divergence block
        elif kl_layerwise:
            map_color = 'Cividis'

            pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
            pred_probs_full = np.zeros_like(layer_probs)
            np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
            
            kl_layerwise_values = compute_kl_vs_previous_layer(layer_probs)
            value_matrix = expand_transition_metric_for_heatmap(kl_layerwise_values, len(layer_probs), layer_probs.shape[1])

            metric_type = 'lw_kl'
            title = f"Layer-wise KL Div ({'mean topk' if topk_mean else 'top-1'})"

        # Token Variety block
        elif token_variety:
            map_color = 'Purples'

            pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)
            pred_probs_full = np.zeros_like(layer_probs)
            np.put_along_axis(pred_probs_full, layer_preds[..., None], pred_probs, axis=-1)
            
            token_variety_values = compute_token_variety(topk_indices)
            value_matrix = expand_metric_for_heatmap(token_variety_values, layer_probs.shape[1])

            metric_type = 'tok_var'
            title = f"Token Variety ({'mean topk' if topk_mean else 'top-1'})"

        # Ranks
        elif ranks:
            map_color = 'Blues'

            if topk_mean:
                # Get ranks for top-k tokens and average over them
                topk_probs = np.take_along_axis(layer_probs, topk_indices, axis=-1)  # shape (L, T, k)
                ranks_matrix = (layer_probs[..., None] >= topk_probs[:, :, None, :]).sum(axis=-2)  # (L, T, k)
                value_matrix = ranks_matrix.mean(axis=-1)  # Average over top-k (L, T)
            else:
                pred_probs = np.take_along_axis(layer_probs, layer_preds[..., None], axis=-1)  # shape (L, T, 1)
                value_matrix = (layer_probs >= pred_probs).sum(axis=-1)  # Compare and sum (L, T)

            # Calculate ranks based on probabilities, sorting in descending order
            rank_matrix_raw = np.argsort(-layer_probs, axis=-1)  # (L, T, Vocab)

            # Calculate the rank of the true predicted token for each layer/token
            # For each token, find its position in the sorted list of probabilities
            pred_ranks = np.take_along_axis(rank_matrix_raw, layer_preds[..., None], axis=-1).squeeze(-1) + 1  # +1 to make rank start from 1

            metric_type = 'ranks'
            title = f"Prediction Rank ({'mean topk' if topk_mean else 'top-1'})"

        # Logits
        else:
            map_color = 'thermal'
            topk_logits = np.take_along_axis(layer_logits, topk_indices, axis=-1)
            if topk_mean:
                value_matrix = topk_logits.mean(axis=-1)
            else:
                value_matrix = np.take_along_axis(layer_logits, layer_preds[..., None], axis=-1).squeeze(-1)
            metric_type = 'logits'
            title = f"Logits ({'mean topk' if topk_mean else 'top-1'})"

        # ---- plot logit lens
        fig = _topk_logit_lens_fig(
            layer_logits=layer_logits,
            layer_preds=layer_preds,
            layer_probs=layer_probs,
            topk_scores=topk_scores,
            topk_indices=topk_indices,
            tokenizer=tokenizer,
            input_ids=input_ids,
            start_ix=start_ix,
            layer_names=layer_names,
            top_k=topk,
            topk_mean=topk_mean,
            normalize=True,
            metric_type=metric_type,
            map_color=map_color,
            value_matrix=value_matrix,
            rank_matrix_raw=rank_matrix_raw,
            pred_ranks=pred_ranks,  
            title=title,
            block_step=block_step,
            token_font_size=token_font_size,
            label_font_size=label_font_size
        )

        clear_cuda_cache()
        return fig
    
    # ---- if batch analysis, logging
    else:
        inputs = [
            # Language understanding
            "The quick brown fox jumps over the lazy dog.",
            "Despite the rain, the event continued as planned.",
            
            # Logic/reasoning
            "If all humans are mortal and Socrates is a human, then Socrates is mortal.",
            "Either the lights are off or the power is out. The lights are on, so the power must be out.",

            # Math/numerical
            "The derivative of sin(x) with respect to x is cos(x).",
            "What is the sum of the first 100 natural numbers?",

            # Programming
            "In Python, list comprehensions provide a concise way to create lists.",
            "To define a function in JavaScript, use the 'function' keyword.",

            # Commonsense knowledge
            "You should refrigerate milk after opening it to keep it fresh.",
            "People usually eat breakfast in the morning before starting their day.",

            # Scientific knowledge
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
            "Photosynthesis is the process by which plants convert sunlight into chemical energy."
        ]
        # Collect metrics if needed
        if json_log_path is not None:
            metrics = collect_logit_lens_metrics_batch(
                model, tokenizer, prompts=inputs, start_ix=start_ix, end_ix=end_ix, topk=topk, prompt_type='text', max_prompts=50
            )

        save_metrics_to_json(metrics, json_log_path)
        clear_cuda_cache()