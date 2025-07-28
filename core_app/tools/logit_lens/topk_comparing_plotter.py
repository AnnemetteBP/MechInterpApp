from functools import partial
from typing import Tuple, List, Dict, Any, Union, Optional
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import matplotlib as mpl
import scipy.special
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
#import colorcet  # noqa
import random
from functools import lru_cache
import plotly.graph_objects as go
from ..util.python_utils import make_print_if_verbose

from .hooks import make_lens_hooks
from .layer_names import make_layer_names


# ---------------------------
# Configs, Model Loading
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

# cache so not re-load on every callback
@lru_cache(maxsize=2)
def _load_model_tokenizer(model_id:str, tok_id:str, quant_config:str|None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok   = AutoTokenizer .from_pretrained(tok_id, trust_remote_code=True)

    if quant_config:
        if '4' in quant_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True if quant_config == 'ptsq4bit' else False,
                bnb_4bit_quant_type='nf4',       # or 'fp4'
                #bnb_4bit_compute_dtype='float16'
            )
        elif '8' in quant_config:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                #bnb_4bit_compute_dtype='float16'
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
def safe_cast_logits(tensor:torch.Tensor) -> torch.Tensor:
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8]:
        tensor = tensor.to(torch.float32)
    return torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)

def numpy_safe_cast(x:Any) -> Any:
    x = x.astype(np.float32)
    return np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)



# ---------------------------
# Tokenize Inputs
# ---------------------------
def text_to_input_ids(tokenizer:Any, text:Union[str, List[str]], model:Optional[torch.nn.Module]=None, add_special_tokens:bool=True, pad_to_max_length=False) -> torch.Tensor:
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
        device = next(model.parameters()).device
        tokens = tokens.to(device)

    return tokens  # shape: [batch_size, seq_len]



# ---------------------------
# Collect & Preprocess Logits
# ---------------------------
def collect_logits(model, input_ids, layer_names, decoder_layer_names=None):
    model._last_resid = None

    # Force the model to store logits via hooks during the forward pass
    with torch.no_grad():
        _ = model(input_ids, output_hidden_states=True)

    model._last_resid = None

    logits_by_layer = []
    for name in layer_names:
        layer_output = model._layer_logits.get(name)
        if layer_output is None:
            raise ValueError(f"Missing logits for layer: {name}")

        # Ensure it's a NumPy array
        if isinstance(layer_output, torch.Tensor):
            layer_output = layer_output.detach().cpu().numpy()

        # Remove batch dim if present
        if layer_output.ndim == 3 and layer_output.shape[0] == 1:
            layer_output = layer_output[0]  # shape: (seq_len, vocab_size)

        logits_by_layer.append(layer_output)

    # Final shape: (num_layers, seq_len, vocab_size)
    layer_logits = np.stack(logits_by_layer, axis=0)

    return layer_logits, layer_names


def postprocess_logits_topk(
    layer_logits: np.ndarray,
    normalize_probs: bool = False,
    top_n: int = 5,
    return_scores: bool = True,
):
    """
    Inputs:
        layer_logits: np.ndarray of shape [L, T, V] or [L, B, T, V]
    Outputs:
        layer_preds: int array [L, T] or [L, B, T]
        layer_probs: float array [L, T, V] or [L, B, T, V]
        top_n_scores: [L, T] or [L, B, T] (mean of top_n probs)
    """
    if isinstance(layer_logits, torch.Tensor):
        layer_logits = layer_logits.cpu().numpy()

    layer_logits = layer_logits.astype(np.float32)
    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    layer_probs = scipy.special.softmax(layer_logits, axis=-1)
    layer_probs = np.nan_to_num(layer_probs, nan=1e-10, posinf=1.0, neginf=0.0)

    if normalize_probs:
        sum_probs = np.sum(layer_probs, axis=-1, keepdims=True)
        sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)
        layer_probs = layer_probs / sum_probs

    layer_preds = layer_probs.argmax(axis=-1)

    if top_n == 1:
        top_n_scores = np.max(layer_probs, axis=-1)
    else:
        sorted_probs = np.sort(layer_probs, axis=-1)
        top_n_scores = np.mean(sorted_probs[..., -top_n:], axis=-1)

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs


def top1_to_onehot(preds:np.ndarray, vocab_size:int) -> Any:
    one_hot = np.zeros((*preds.shape, vocab_size), dtype=np.float32)
    it = np.nditer(preds, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        one_hot[idx][preds[idx]] = 1.0
        it.iternext()
    
    return one_hot



# ---------------------------
# Correctness / Accuracy & Stability
# ---------------------------
def compute_correctness_metrics(layer_preds:Any, target_ids:Any, topk_indices=None) -> Tuple[Any,Any|None]:
    correct_1 = (layer_preds[-1] == target_ids).astype(int)

    if topk_indices is not None:
        # target_ids: [tokens], reshape to [1, tokens, 1] to broadcast against [layers, tokens, topk]
        target_ids_broadcasted = target_ids[None, :, None]
        correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)
    else:
        correct_topk = None

    return correct_1, correct_topk


def accuracy(preds:Any, targets:Any, topn:Any) -> Tuple[Any,Any,Any,Any]:
    top1 = preds[..., 0]
    acc1 = (top1[:, :-1] == targets).astype(int)
    acc_top1 = acc1.mean()

    in_topn = np.any(preds[:, :-1] == targets[..., None], axis=-1).astype(int)
    acc_topn = in_topn.mean()

    return acc_top1, acc_topn, acc1.sum(), in_topn.sum()

def stability(preds:Any, targets:Any) -> np.ndarray:
    seq_len = preds.shape[1]
    num_layers = preds.shape[0]
    stab = np.full(seq_len, fill_value=num_layers)
    
    for t in range(seq_len):
        for l in range(num_layers):
            if preds[l, t] == targets[0, t]:
                stab[t] = l
                break
    
    return stab



# ---------------------------
# Metrics & Clipping Helpers
# ---------------------------
def compute_kl_divergence(logits1:Any, logits2:Any) -> Any:
    probs1 = scipy.special.softmax(logits1, axis=-1)
    probs2 = scipy.special.softmax(logits2, axis=-1)
    return np.sum(kl_div(probs1, probs2), axis=-1)

def get_value_at_preds(values:Any, preds:Any) -> Any:
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x:Any, tokenizer:Any, quotemark="") -> Any:
    return quotemark + str(tokenizer.decode([x])) + quotemark

def compute_entropy(probs:Any) -> Any:
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def maybe_batchify(p:Any) -> Any:
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p

def min_max_scale(arr:Any, new_min:Any, new_max:Any) -> Any:
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

def clipmin(x:Any, clip:Any) -> Any:
    return np.clip(x, a_min=clip, a_max=None)

def kl_summand(p:Any, q:Any, clip=1e-16) -> Any:
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)

def kl_divergence(p:Any, q:Any, axis=-1, clip=1e-16) -> Any:
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)

def js_divergence(p:Any, q:Any, axis=-1, clip=1e-16) -> Any:
    """Computes Jensen-Shannon divergence between two probability distributions."""
    p, q = clipmin(p, clip), clipmin(q, clip)
    m = (p + q) / 2
    return (kl_divergence(p, m, axis=axis, clip=clip) + kl_divergence(q, m, axis=axis, clip=clip)) / 2

def comput_entropy(p:Any, axis=-1, clip=1e-16) -> Any:
    p = clipmin(p, clip)
    return -np.sum(p * np.log(p), axis=axis)

def nwd(p:Any, q:Any, axis=-1, clip=1e-16) -> Any:
    """Computes normalized Wasserstein distance between two probability distributions."""
    p, q = np.clip(p, clip, 1.0), np.clip(q, clip, 1.0)

    # Ensure both are 3D: (batch, seq_len, vocab)
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    if q.ndim == 2:
        q = np.expand_dims(q, 0)

    # Align batch sizes
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.repeat(p, q.shape[0], axis=0)
    elif q.shape[0] == 1 and p.shape[0] > 1:
        q = np.repeat(q, p.shape[0], axis=0)

    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch after fixing: p {p.shape}, q {q.shape}")

    vocab_size = p.shape[-1]
    indices = np.arange(vocab_size)

    distances = np.array([
        wasserstein_distance(indices, indices, p[b, i], q[b, i])
        for b in range(p.shape[0])
        for i in range(p.shape[1])
    ])

    # Reshape back to (batch, seq_len)
    distances = distances.reshape(p.shape[0], p.shape[1])
    return distances / vocab_size



# ---------------------------
# Metrics & Computation Configs
# ---------------------------
METRIC_REGISTRY = {
    "js":            {"type": "prob",  "topk": True,  "title": "Jensen–Shannon divergence", "cmap": "Blues"},
    "nwd":           {"type": "prob",  "topk": True,  "title": "Normalized Wasserstein distance (top-k)", "cmap": "Blues"},
    "full_nwd":      {"type": "prob",  "topk": False, "title": "Normalized Wasserstein distance (full)", "cmap": "Blues"},
    "kl":            {"type": "prob",  "topk": False, "title": "KL Divergence (P||Q)", "cmap": "Reds"},
    "cosine":        {"type": "prob",  "topk": False, "title": "Cosine similarity", "cmap": "Viridis"},
    "entropy_diff":  {"alias": "entropy_gap", "type": "prob", "topk": False, "title": "Entropy difference (P - Q)", "cmap": "RdBu"},

    "match_ratio":   {"alias": "agreement", "type": "top1", "topk": False, "title": "Top-1 agreement", "cmap": "Cividis"},
    "rank_delta":    {"type": "top1", "topk": False, "title": "Top-1 rank delta", "cmap": "Cividis"},
    "jaccard":       {"type": "topk", "topk": True,  "title": "Top-k Jaccard", "cmap": "Cividis"},
    "variety":       {"type": "top1", "topk": False, "title": "Layer token variety", "cmap": "Cividis"},
}


def resolve_metric_type(metric_type:str) -> Any:
    info = METRIC_REGISTRY.get(metric_type)
    if info is None:
        raise ValueError(f"Unsupported metric: {metric_type}")
    return info.get('alias', metric_type), info


def add_batch_dim(arr:np.ndarray) -> np.ndarray:
    """
    Add batch dimension if missing:
    Accepts shapes (L, T), (L, T, V), or (L, T) without batch dim,
    returns array with batch dim: (L, 1, T) or (L, 1, T, V) etc.

    Args:
        arr: np.ndarray

    Returns:
        np.ndarray with batch dim added at axis=1
    """
    if arr.ndim == 3:  # e.g. (L, T, V)
        return np.expand_dims(arr, axis=1)  # (L, 1, T, V)
    elif arr.ndim == 2:  # e.g. (L, T)
        return np.expand_dims(arr, axis=1)  # (L, 1, T)
    elif arr.ndim == 4 or arr.ndim == 3:
        return arr
    else:
        raise ValueError(f"Unexpected ndim={arr.ndim} for array shape {arr.shape}")

def strip_batch_dim(arr:np.ndarray) -> np.ndarray:
    """
    Strip batch dimension if it exists: from (L, 1, T) → (L, T)
    """
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    return arr


def compute_comparison_metric(
    metric_type:str,
    layer_probs_1:np.ndarray,  # [B, T, V]
    layer_probs_2:np.ndarray,  # [B, T, V]
    layer_preds_1:np.ndarray,  # [B, T]
    layer_preds_2:np.ndarray,  # [B, T]
    topk: int
) -> np.ndarray:
    """
    Returns: [B, T]
    """
    metric_type, meta = resolve_metric_type(metric_type)

    if meta['topk'] and (topk is None or topk <= 0):
        raise ValueError(f"Metric '{metric_type}' requires top-k > 0.")

    B, T, V = layer_probs_1.shape

    if metric_type == 'js':
        # Take top-k of model_2 to align supports
        topk_idx = np.argsort(layer_probs_2, axis=-1)[..., -topk:]
        p1 = np.take_along_axis(layer_probs_1, topk_idx, axis=-1)
        p2 = np.take_along_axis(layer_probs_2, topk_idx, axis=-1)
        p1 = p1 / np.clip(p1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        p2 = p2 / np.clip(p2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        return js_divergence(p1, p2, axis=-1)

    elif metric_type == 'nwd':
        topk_idx = np.argsort(layer_probs_1 + layer_probs_2, axis=-1)[..., -topk:]
        p1 = np.take_along_axis(layer_probs_1, topk_idx, axis=-1)
        p2 = np.take_along_axis(layer_probs_2, topk_idx, axis=-1)
        p1 = p1 / np.clip(p1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        p2 = p2 / np.clip(p2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        return nwd(p1, p2)

    elif metric_type == 'full_nwd':
        p1 = layer_probs_1 / np.clip(layer_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        p2 = layer_probs_2 / np.clip(layer_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        return nwd(p1, p2)

    elif metric_type == 'kl':
        p = layer_probs_1 / np.clip(layer_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        q = layer_probs_2 / np.clip(layer_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
        return kl_divergence(p, q, axis=-1)

    elif metric_type == 'entropy_gap':
        H1 = comput_entropy(layer_probs_1, axis=-1)
        H2 = comput_entropy(layer_probs_2, axis=-1)
        return H1 - H2

    elif metric_type == 'agreement':
        return (layer_preds_1 == layer_preds_2).astype(np.float32)

    elif metric_type == 'jaccard':
        # set-based over top-k indices
        topk_1 = np.argsort(layer_probs_1, axis=-1)[..., -topk:]
        topk_2 = np.argsort(layer_probs_2, axis=-1)[..., -topk:]
        out = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                s1 = set(topk_1[b, t].tolist())
                s2 = set(topk_2[b, t].tolist())
                inter = len(s1 & s2)
                uni = len(s1 | s2)
                out[b, t] = inter / max(uni, 1e-6)
        return out

    elif metric_type == 'rank_delta':
        # ranks: lower is better
        ranks_1 = np.argsort(np.argsort(-layer_probs_1, axis=-1), axis=-1)
        ranks_2 = np.argsort(np.argsort(-layer_probs_2, axis=-1), axis=-1)
        # use model_1 top1 indices to probe rank difference
        idx = layer_preds_1[..., None]
        r1 = np.take_along_axis(ranks_1, idx, axis=-1).squeeze(-1)
        r2 = np.take_along_axis(ranks_2, idx, axis=-1).squeeze(-1)
        return np.abs(r1 - r2).astype(np.float32)

    elif metric_type == 'cosine':
        dot = np.sum(layer_probs_1 * layer_probs_2, axis=-1)
        n1 = np.linalg.norm(layer_probs_1, axis=-1)
        n2 = np.linalg.norm(layer_probs_2, axis=-1)
        return dot / np.clip(n1 * n2, 1e-10, None)

    elif metric_type == 'variety':
        # how many unique top-1 preds appeared across layers is not computable per single layer (needs all layers)
        # compute per position after stacking in get_metric_matrix, so here just returning zeros (placeholder)
        return np.zeros((B, T), dtype=np.float32)

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def get_metric_matrix(
    metric_type:str,
    all_layer_probs_1:np.ndarray,  
    all_layer_probs_2:np.ndarray,  
    all_layer_preds_1:np.ndarray,  
    all_layer_preds_2:np.ndarray,  
    topk:int=None,
    agg_mode:str='none'  # "layer", "position", or "none"
) -> np.ndarray:
    """
    Compute metric matrix with shape [L, T].
    agg_mode defines how to aggregate over batch/layer/position before final shape.
    """

    L, B, T = all_layer_probs_1.shape[:3]

    # Compute per-layer [B, T] metrics
    vals = []
    for l in range(L):
        v = compute_comparison_metric(
            metric_type,
            all_layer_probs_1[l],  
            all_layer_probs_2[l],
            all_layer_preds_1[l], 
            all_layer_preds_2[l],
            topk
        )  # [B, T]
        vals.append(v)

    metric_matrix = np.stack(vals, axis=0)  # [L, B, T]

    if metric_type == 'variety':
        # Custom handling for variety: # of unique preds at each position
        unique_counts = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                unique_counts[b, t] = len(np.unique(all_layer_preds_2[:, b, t]))
        unique_counts /= L
        metric_matrix = np.broadcast_to(unique_counts[None, ...], (L, B, T)).copy()

    # Aggregate according to mode
    """if agg_mode == "none":
        # Average over batch → [L, T]
        return metric_matrix.mean(axis=1)

    elif agg_mode == "layer":
        # Avg over layers and batch → [T]
        avg = metric_matrix.mean(axis=0).mean(axis=0)  # [T]
        return np.tile(avg[None, :], (L, 1))           # Broadcast across layers

    elif agg_mode == "position":
        # Avg over positions and batch → [L]
        avg = metric_matrix.mean(axis=2).mean(axis=1)  # [L]
        return np.tile(avg[:, None], (1, T))           # Broadcast across positions

    else:
        raise ValueError(f"Invalid agg_mode: {agg_mode}")"""
    if agg_mode == 'none':
        # average over batch only → shape [L, T]
        return metric_matrix.mean(axis=1)

    elif agg_mode == 'layer':
        # average over batch and tokens → shape [L]
        avg = metric_matrix.mean(axis=(1, 2))
        return np.tile(avg[:, None], (1, T))

    elif agg_mode == 'position':
        # average over batch and layers → shape [T]
        avg = metric_matrix.mean(axis=(0, 1))
        return np.tile(avg[None, :], (L, 1))

    else:
        raise ValueError(f"Invalid agg_mode: {agg_mode}")



# ---------------------------
# Plotting
# ---------------------------
def mm_scale(arr:Any, new_min:int=0, new_max:int=1) -> Any:
    arr = np.array(arr)
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max == arr_min:
        return np.full_like(arr, (new_min + new_max) / 2)
    
    return (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min


def _topk_comparing_lens_fig(
    layer_logits:Any,       
    layer_preds:Any,      
    layer_probs:Any,        
    topk_scores:Any,  
    topk_indices:Any,       
    tokenizer:Any,
    input_ids:Any,         
    start_ix:int,
    layer_names:List[str],
    top_k:int=5,
    normalize:bool=True,
    metric_type:Optional[str]=None,
    value_matrix:Optional[np.ndarray]=None,  # [L, T]
    title:Optional[str]=None,
    map_color:Optional[str]=None,
    block_step:int=1,
    token_font_size:int=12,
    label_font_size:int=20,
    agg_mode:str='none',
) -> go.Figure:

    if metric_type is not None and metric_type in METRIC_REGISTRY:
        reg = METRIC_REGISTRY[metric_type]
        if title is None:
            title = reg.get('title', metric_type)
        if map_color is None:
            map_color = reg.get('cmap', 'Cividis')
    else:
        title = title or "Top-k mean prob"
        map_color = map_color or 'Cividis'

    L, T, V = layer_logits.shape
    end_ix = start_ix + T

    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < T:
        next_token_text.append("")

    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)
    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    assert value_matrix is not None, "value_matrix must be provided"
    assert value_matrix.shape == (L, T), f"value_matrix must be [L,T], got {value_matrix.shape}"

    # normalize before plotting
    if normalize:
        value_matrix = mm_scale(value_matrix, 0, 1)

    """if agg_mode == "position":
        # metric varies per position → vertical stripes → same value down column
        value_matrix = np.tile(value_matrix[:, 0][:, None], (1, T))
    elif agg_mode == "layer":
        # metric varies per layer → horizontal stripes → same value across row
        value_matrix = np.tile(value_matrix[0][None, :], (L, 1))"""

    # Subsample layers
    keep_idxs = [0] + list(range(1, L - 1, block_step)) + [L - 1]
    pred_token_text = np.asarray(pred_token_text)[keep_idxs]
    topk_tokens = np.asarray(topk_tokens)[keep_idxs]
    value_matrix = value_matrix[keep_idxs]
    layer_names = [layer_names[i] for i in keep_idxs]
    
    # Align correctness mask
    true_next_tokens = np.array(next_token_text[:T])
    pred_tokens_layer = pred_token_text[:, :T]
    is_correct = (pred_tokens_layer == true_next_tokens[None, :])

    input_tokens_matrix = np.tile(input_tokens_str, (pred_tokens_layer.shape[0], 1))
    echo_mask = (pred_tokens_layer == input_tokens_matrix)

    for j in range(echo_mask.shape[1]):
        for i in range(1, echo_mask.shape[0]):
            if not echo_mask[i - 1, j]:
                echo_mask[i:, j] = False
                break
    is_correct = is_correct & ~echo_mask

    # Build hover text
    hovertext = []
    for i in range(len(layer_names)):
        row = []
        for j in range(T):
            val = value_matrix[i, j]
            if metric_type is not None:
                hv = f"<b>{metric_type}</b>: {val:.4f}<br>"
            else:
                hv = f"<b>Score</b>: {val:.4f}<br>"
            hv += f"<b>Pred:</b> {pred_token_text[i, j]}<br><b>Top-{top_k}:</b><br>"
            for k in range(min(top_k, topk_tokens.shape[-1])):
                hv += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}<br>"
            row.append(hv)
        hovertext.append(row)

    # Reverse for plot alignment (because y-axis is reversed)
    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    is_correct = is_correct[::-1]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(T)),
        y=list(range(len(layer_names))),
        text=pred_token_text,
        texttemplate="%{text}",
        textfont=dict(size=token_font_size),
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        zmin=np.min(value_matrix),
        zmax=np.max(value_matrix),
        showscale=True
    ))

    cy, cx = np.where(is_correct == 1)
    for y, x in zip(cy, cx):
        fig.add_shape(
            type='rect',
            x0=x - 0.5, x1=x + 0.5,
            y0=y - 0.5, y1=y + 0.5,
            line=dict(color='black', width=2),
            layer='above'
        )

    # Overlay scatter for top axis
    fig.add_trace(go.Scatter(
        x=list(range(T)),
        y=[None] * T,
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
            tickvals=list(range(T)),
            ticktext=input_tokens_str,
            title='Input Token',
            side='bottom',
            anchor='y',
            domain=[0.0, 1.0]
        ),
        xaxis2=dict(
            tickmode='array',
            tickvals=list(range(T)),
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
        width=max(1200, 100 * T),
        height=max(600, 35 * len(layer_names)),
        margin=dict(l=20, r=10, t=40, b=10),
        title=title
    )

    return fig



# ---------------------------
# Comparing (Logit) Lens Plotter
# ---------------------------
def plot_topk_comparing_lens(
    model_1:Any,
    model_2:Any,
    tokenizer_1:Any,
    tokenizer_2:Any,
    inputs:Union[str, List[str], None],
    start_ix:int,
    end_ix:int,
    topk:int=5,
    topk_mean:bool=True,
    metric_type:Optional[str]=None,
    agg_mode:str='none',  # "none", "layer", "position"
    block_step:int=1,
    token_font_size:int=12,
    label_font_size:int=20,
    include_input:bool=True,
    force_include_output:bool=True,
    include_subblocks:bool=False,
    decoder_layer_names:List[str] = ['norm', 'lm_head'],
    top_down:bool=False,
    verbose:bool=False,
    pad_to_max_length:bool=False,
    model_precision_1:Optional[str]=None,
    model_precision_2:Optional[str]=None,
    use_deterministic_backend:bool=False
) -> go.Figure:
    """ Plots the Comparing (Logit) Lens for topk """

    topk = 1 if topk < 1 else topk
    topk_mean = False if topk == 1 else topk_mean

    if isinstance(inputs, str):
        inputs = [inputs]
    elif inputs is None:
        inputs = ["What is y if y=2*2-4+(3*2)"]

    # ---- load models
    model_1, tokenizer_1 = _load_model_tokenizer(model_1, tokenizer_1, model_precision_1)
    model_2, tokenizer_2 = _load_model_tokenizer(model_2, tokenizer_2, model_precision_2)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    if use_deterministic_backend:
        _set_deterministic_backend()

    # ---- layer names
    layer_names_1 = make_layer_names(model_1, block_step, include_input, force_include_output, include_subblocks, decoder_layer_names)
    layer_names_2 = make_layer_names(model_2, block_step, include_input, force_include_output, include_subblocks, decoder_layer_names)

    # ---- hooks
    make_lens_hooks(model_1, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_1,
                    decoder_layer_names=decoder_layer_names, verbose=verbose)
    make_lens_hooks(model_2, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_2,
                    decoder_layer_names=decoder_layer_names, verbose=verbose)

    # ---- inputs
    input_ids_1 = text_to_input_ids(tokenizer_1, inputs, model_1, pad_to_max_length=pad_to_max_length)
    input_ids_2 = text_to_input_ids(tokenizer_2, inputs, model_2, pad_to_max_length=pad_to_max_length)

    input_ids_1 = input_ids_1.to(next(model_1.parameters()).device)
    input_ids_2 = input_ids_2.to(next(model_2.parameters()).device)

    seq_len = input_ids_1.shape[1]
    if end_ix > seq_len:
        if verbose:
            print(f"Adjusting end_ix from {end_ix} to {seq_len}")
        end_ix = seq_len
    if start_ix >= seq_len or start_ix < 0:
        raise ValueError(f"start_ix {start_ix} is out of range for sequence length {seq_len}")
    if start_ix >= end_ix:
        raise ValueError(f"start_ix {start_ix} must be less than end_ix {end_ix}")

    # ---- collect logits
    layer_logits_1, _ = collect_logits(model_1, input_ids_1, layer_names_1)  # [L, T, V]
    layer_logits_2, _ = collect_logits(model_2, input_ids_2, layer_names_2)

    layer_logits_1 = safe_cast_logits(torch.tensor(layer_logits_1)).cpu().numpy()
    layer_logits_2 = safe_cast_logits(torch.tensor(layer_logits_2)).cpu().numpy()

    layer_logits_1 = layer_logits_1[:, start_ix:end_ix, :]
    layer_logits_2 = layer_logits_2[:, start_ix:end_ix, :]

    # ---- postprocess (keep your original function AS IS)
    if topk_mean:
        layer_preds_1, layer_probs_1, _ = postprocess_logits_topk(layer_logits_1, top_n=topk, return_scores=True)
        layer_preds_2, layer_probs_2, _ = postprocess_logits_topk(layer_logits_2, top_n=topk, return_scores=True)
    else:
        layer_preds_1, layer_probs_1 = postprocess_logits_topk(layer_logits_1, top_n=1, return_scores=False)
        layer_preds_2, layer_probs_2 = postprocess_logits_topk(layer_logits_2, top_n=1, return_scores=False)

    # ---- fix NaNs
    layer_probs_1 = np.nan_to_num(layer_probs_1, nan=1e-10, posinf=1.0, neginf=0.0)
    layer_probs_2 = np.nan_to_num(layer_probs_2, nan=1e-10, posinf=1.0, neginf=0.0)

    # ---- add batch dim (L, 1, T)
    layer_probs_1_b = add_batch_dim(layer_probs_1)
    layer_probs_2_b = add_batch_dim(layer_probs_2)
    layer_preds_1_b = add_batch_dim(layer_preds_1)
    layer_preds_2_b = add_batch_dim(layer_preds_2)

    value_matrix_3d = get_metric_matrix(
        metric_type=metric_type,
        all_layer_probs_1=layer_probs_1_b,
        all_layer_probs_2=layer_probs_2_b,
        all_layer_preds_1=layer_preds_1_b,
        all_layer_preds_2=layer_preds_2_b,
        topk=topk,
        #agg_mode='none'
        agg_mode=agg_mode
    )

    value_matrix = value_matrix_3d

    # ---- get top-k indices for hover
    topk_indices = None
    if topk is not None and topk > 0:
        if layer_probs_2.ndim >= 3:
            topk_indices = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]
        elif layer_probs_2.ndim == 2:
            topk_indices = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]

    # ---- plot
    fig = _topk_comparing_lens_fig(
        layer_logits=layer_logits_2,
        layer_preds=layer_preds_2,
        layer_probs=layer_probs_2,
        topk_scores=None,
        topk_indices=topk_indices,
        tokenizer=tokenizer_2,
        input_ids=input_ids_2,
        start_ix=start_ix,
        layer_names=layer_names_2,
        top_k=topk,
        normalize=True,
        metric_type=metric_type,
        value_matrix=value_matrix,
        title=None,
        map_color=None,
        block_step=block_step,
        token_font_size=token_font_size,
        label_font_size=label_font_size,
        agg_mode=agg_mode, 
    )

    clear_cuda_cache()
    return fig