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

# ===================== Clear Cache ============================
def clear_cuda_cache():
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()

def safe_cast_logits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in [torch.float16, torch.bfloat16, torch.int8, torch.uint8]:
        tensor = tensor.to(torch.float32)
    return torch.nan_to_num(tensor, nan=-1e9, posinf=1e9, neginf=-1e9)

def numpy_safe_cast(x):
    x = x.astype(np.float32)
    return np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9)

# cache so we don’t re-load on every callback
@lru_cache(maxsize=2)
def _load_model_tokenizer(model_id:str, tok_id:str, quant_config:str|None):
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
# ===================== Tokenize input texts ============================
def text_to_input_ids(tokenizer: Any, text: Union[str, List[str]], model: Optional[torch.nn.Module] = None, add_special_tokens: bool = True, pad_to_max_length=False) -> torch.Tensor:
    """
    Tokenize the inputs, respecting padding behavior.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure EOS token is used if padding is missing

    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Padding to the longest sequence in the batch or to max length
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest" if not pad_to_max_length else True,  # Padding only to longest sequence
        truncation=True,
        add_special_tokens=add_special_tokens,
    )["input_ids"]

    if model is not None:
        device = next(model.parameters()).device
        tokens = tokens.to(device)

    return tokens  # shape: [batch_size, seq_len]

# ===================== Layer Logits ============================
"""def collect_logits(model, input_ids, layer_names, decoder_layer_names:Optional[List|str|None]=None):
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
    
    # Ensure we gather logits for each layer (whether for single or batch)
    try:
        layer_logits = np.concatenate(
            [model._layer_logits.get(name, np.zeros((batch_size, model.config.hidden_size))) for name in layer_names],
            axis=0,
        )
    except KeyError as e:
        print(f"[Error] Missing layer logits for {e}")
        layer_logits = np.zeros((batch_size, len(layer_names), model.config.hidden_size))

    return layer_logits, layer_names"""

def collect_hidden_states(model, input_ids, layer_names):
    """
    Extract hidden states for specified layers for input_ids.
    Returns:
        hidden_states: np.ndarray with shape (batch_size, num_layers, seq_len, hidden_size)
        layer_names: list of layers in order
    """
    model._last_resid = None  # reset cache if used

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    all_hidden_states = outputs.hidden_states  # tuple: one tensor per layer (including embeddings)

    hidden_states_list = []
    for name in layer_names:
        # Get index for this layer name — example assuming "layer.{i}"
        idx = int(name.split('.')[-1]) + 1  # +1 because hidden_states[0] = embedding layer
        hs = all_hidden_states[idx]  # tensor: (batch_size, seq_len, hidden_size)
        hidden_states_list.append(hs.cpu().numpy())

    # Stack layers: shape (batch_size, num_layers, seq_len, hidden_size)
    hidden_states = np.stack(hidden_states_list, axis=1)

    return hidden_states, layer_names


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


# ===================== Probs and logits for topk > 1 and for topk plot ============================
"""def postprocess_logits_topk(layer_logits: Any, normalize_probs=False, top_n:int=5, return_scores:bool=True) -> Tuple[Any, Any, Any]:

    if layer_logits.dtype == np.float16:
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

    #print(f"[DEBUG PROBS] {layer_probs} | ")

    # Get the index of the maximum logit (predicted token)
    layer_preds = layer_logits.argmax(axis=-1)

    # Compute the mean of the top-N probabilities for each token
    top_n_scores = np.mean(
        np.sort(layer_probs, axis=-1)[:, -top_n:], axis=-1
    )

    if return_scores:
        return layer_preds, layer_probs, top_n_scores
    else:
        return layer_preds, layer_probs"""

def postprocess_logits_topk(layer_logits:np.ndarray, normalize_probs=False, top_n:int=5, return_scores:bool=True):
    if isinstance(layer_logits, torch.Tensor):
        layer_logits = layer_logits.cpu().numpy()

    if layer_logits.dtype == np.float16:
        layer_logits = layer_logits.astype(np.float32)

    layer_logits = np.nan_to_num(layer_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

    # shape: (batch, seq_len, vocab)
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


def top1_to_onehot(preds: np.ndarray, vocab_size: int):
    one_hot = np.zeros((*preds.shape, vocab_size), dtype=np.float32)
    it = np.nditer(preds, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        one_hot[idx][preds[idx]] = 1.0
        it.iternext()
    return one_hot

# ===================== Correctness ============================
def compute_correctness_metrics(layer_preds, target_ids, topk_indices=None):
    correct_1 = (layer_preds[-1] == target_ids).astype(int)

    if topk_indices is not None:
        # target_ids: [tokens], reshape to [1, tokens, 1] to broadcast against [layers, tokens, topk]
        target_ids_broadcasted = target_ids[None, :, None]
        correct_topk = np.any(topk_indices == target_ids_broadcasted, axis=-1).astype(int)
    else:
        correct_topk = None

    return correct_1, correct_topk

# ===================== Clipping and metric helpers ============================
def compute_kl_divergence(logits1, logits2):
    probs1 = scipy.special.softmax(logits1, axis=-1)
    probs2 = scipy.special.softmax(logits2, axis=-1)
    return np.sum(kl_div(probs1, probs2), axis=-1)

def compute_wasserstein_from_json(file_a, file_b, key="logit_mean"):
    with open(file_a) as f1, open(file_b) as f2:
        m1 = json.load(f1)[key]
        m2 = json.load(f2)[key]
    return wasserstein_distance(m1, m2)

def get_value_at_preds(values, preds):
    return np.stack([values[:, j, preds[j]] for j in range(preds.shape[-1])], axis=-1)

def num2tok(x, tokenizer, quotemark=""):
    return quotemark + str(tokenizer.decode([x])) + quotemark

def compute_entropy(probs):
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=-1)

def maybe_batchify(p):
    """ Normalize shape for e.g., wasserstein """
    if p.ndim == 2:
        p = np.expand_dims(p, 0)
    return p

def min_max_scale(arr, new_min, new_max):
    """Scales an array to a new range [new_min, new_max] for plotting and comparison of normalized values across models."""
    old_min, old_max = np.min(arr), np.max(arr)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min if old_max > old_min else arr

def clipmin(x, clip):
    return np.clip(x, a_min=clip, a_max=None)

def kl_summand(p, q, clip=1e-16):
    p, q = clipmin(p, clip), clipmin(q, clip)
    return p * np.log(p / q)

def kl_divergence(p, q, axis=-1, clip=1e-16):
    return np.sum(kl_summand(p, q, clip=clip), axis=axis)

def js_divergence(p, q, axis=-1, clip=1e-16):
    """Computes Jensen-Shannon divergence between two probability distributions."""
    p, q = clipmin(p, clip), clipmin(q, clip)
    m = (p + q) / 2
    return (kl_divergence(p, m, axis=axis, clip=clip) + kl_divergence(q, m, axis=axis, clip=clip)) / 2

def nwd(p, q, axis=-1, clip=1e-16):
#def nwd(p, q, axis=-1, clip=1e-16):
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
    #print(f"[Distances] {distances} | \n[NWD] {distances/vocab_size} |\n")
    return distances / vocab_size

def accuracy(preds, targets, topn):
    top1 = preds[..., 0]
    acc1 = (top1[:, :-1] == targets).astype(int)
    acc_top1 = acc1.mean()

    in_topn = np.any(preds[:, :-1] == targets[..., None], axis=-1).astype(int)
    acc_topn = in_topn.mean()

    return acc_top1, acc_topn, acc1.sum(), in_topn.sum()

def stability(preds, targets):
    seq_len = preds.shape[1]
    num_layers = preds.shape[0]
    stab = np.full(seq_len, fill_value=num_layers)
    for t in range(seq_len):
        for l in range(num_layers):
            if preds[l, t] == targets[0, t]:
                stab[t] = l
                break
    return stab


def _topk_comparing_lens_fig(
    layer_logits,
    layer_preds,
    layer_probs,
    topk_scores,
    topk_indices,
    tokenizer,
    input_ids,
    start_ix,
    layer_names,
    top_k=5,
    normalize=True,
    metric_type:str|None=None,
    map_color='Cividis',
    value_matrix=None,
    title:str|None=None,
    block_step:int=1,
    token_font_size:int=12,
    label_font_size:int=20,
)-> go.Figure:
    

    num_layers, num_tokens, vocab_size = layer_logits.shape

    end_ix = start_ix + num_tokens

    # Decode input and next tokens
    input_token_ids = input_ids[0][start_ix:end_ix]
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_token_ids]

    full_token_ids = input_ids[0]
    next_token_ids = full_token_ids[start_ix + 1:end_ix + 1]
    next_token_text = [tokenizer.decode([tid]) for tid in next_token_ids]
    if len(next_token_text) < num_tokens:
        next_token_text.append("")

    # Decode predictions and top-k tokens
    pred_token_text = np.vectorize(lambda idx: tokenizer.decode([idx]))(layer_preds)
    topk_tokens = np.vectorize(lambda idx: tokenizer.decode([idx]), otypes=[str])(topk_indices)

    # Default: use mean top-k probability if no divergence matrix provided
    if value_matrix is None:
        value_matrix = topk_scores.mean(axis=-1)  # shape: (num_layers, num_tokens)

    print("Shape before any slicing:", value_matrix.shape)

    # Make sure value_matrix is 2D (num_layers, num_tokens)
    if value_matrix.ndim == 3:
        value_matrix = value_matrix.mean(axis=-1)
        print("Shape after mean reduction:", value_matrix.shape)

    # --- FIX START: Define keep_idxs ONCE ---
    keep_idxs = [0] + list(range(1, num_layers - 1, block_step)) + [num_layers - 1]

    # Convert pred_token_text, topk_scores, topk_tokens to numpy arrays if not already
    pred_token_text = np.array(pred_token_text)
    topk_scores = np.array(topk_scores)
    topk_tokens = np.array(topk_tokens)

    # Slice ALL arrays ONCE here, no repeated slicing later!
    value_matrix = value_matrix[keep_idxs]
    pred_token_text = pred_token_text[keep_idxs]
    topk_scores = topk_scores[keep_idxs]
    topk_tokens = topk_tokens[keep_idxs]
    layer_names = [layer_names[i] for i in keep_idxs]

    print("Shape after applying keep_idxs:", value_matrix.shape)

    # Prepare hover text
    hovertext = []
    for i in range(len(layer_names)):
        row = []
        for j in range(num_tokens):
            val = value_matrix[i, j]
            if metric_type == "js":
                hover_val = f"<b>JS Divergence:</b> {val:.3f}<br>"
            elif metric_type == "nwd":
                hover_val = f"<b>NWD:</b> {val:.3f}<br>"
            else:
                hover_val = f"<b>Mean Top-{top_k} Prob:</b> {val:.3f}<br>"
            hover = hover_val + f"<b>Pred:</b> {pred_token_text[i, j]}<br><b>Top-{top_k}:</b><br>"
            for k in range(top_k):
                hover += f"&nbsp;&nbsp;{topk_tokens[i, j, k]}: {topk_scores[i, j, k]:.3f}<br>"
            row.append(hover)
        hovertext.append(row)

    # Normalize values for consistent color range
    if normalize:
        value_matrix = min_max_scale(value_matrix, 0, 1)

    # Decode predictions from selected layers (using keep_idxs already applied)
    # pred_tokens_str is already sliced; no need to slice again
    pred_tokens_str = pred_token_text

    # Ensure next_token_text used for xaxis2 is what we compare against
    true_next_token_text = next_token_text[:num_tokens]  # true next tokens at t+1
    if len(true_next_token_text) < num_tokens:
        true_next_token_text += [""] * (num_tokens - len(true_next_token_text))  # pad if needed

    # Repeat next_token_text across the number of layers to match dimensions
    correct_tokens_matrix = np.tile(true_next_token_text, (pred_tokens_str.shape[0], 1))

    # Compare prediction tokens to correct next tokens
    is_correct = (pred_tokens_str == correct_tokens_matrix)

    # Decode input tokens for the token span
    input_tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0][start_ix:end_ix]]
    input_tokens_matrix = np.tile(input_tokens_str, (pred_tokens_str.shape[0], 1))

    # A mask: for each (layer, token), is this just echoing the input?
    echo_mask = (pred_tokens_str == input_tokens_matrix)

    # For each token position, find whether earlier layers ONLY echoed input
    for j in range(echo_mask.shape[1]):  # loop over token positions
        for i in range(1, echo_mask.shape[0]):
            if not echo_mask[i - 1, j]:
                echo_mask[i:, j] = False
                break

    # Filter: suppress only those that are still pure echoes
    is_correct = is_correct & ~echo_mask

    # Flip for correct top-to-bottom layer visualization
    value_matrix = value_matrix[::-1]
    pred_token_text = pred_token_text[::-1]
    hovertext = hovertext[::-1]
    is_correct = is_correct[::-1]


    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=value_matrix,
        x=list(range(num_tokens)),
        y=list(range(len(layer_names))),
        text=pred_token_text,
        texttemplate="%{text}",
        textfont=dict(size=token_font_size),
        hovertext=hovertext,
        hoverinfo='text',
        colorscale=map_color,
        zmin=np.min(value_matrix),
        zmax=np.max(value_matrix),
    ))

    # Find the correct prediction positions
    correct_y, correct_x = np.where(is_correct[::-1, :] == 1)
    # Mark the predictions that match the tokens on xaxis2 (next_token_text)
    for y, x in zip(correct_y, correct_x):
        adjusted_y = len(layer_names) - 1 - y  # unflip layer index
        fig.add_shape(
            type="rect",
            x0=x - 0.5, x1=x + 0.5,
            y0=adjusted_y - 0.5, y1=adjusted_y + 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )

    # Dummy trace to activate next-token axis
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


def plot_topk_comparing_lens(
    model_1:Any,
    model_2:Any,
    tokenizer_1:Any,
    tokenizer_2:Any,
    inputs:Union[str, List[str], None]|str,
    start_ix:int,
    end_ix:int,
    topk:int=5,
    topk_mean:bool=True,
    js:bool=False,
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
    model_precision_1:Optional[str|None]=None,
    model_precision_2:Optional[str|None]=None,
    use_deterministic_backend:bool=False
) -> go.Figure:

    metric_type = None
    if topk == 1:
        topk_mean = False

    diagnostics = {
        "model_1": {},
        "model_2": {},
        "comparison": {}
    }

    if isinstance(inputs, str):
        inputs = [inputs]
    elif inputs is None:
        inputs = ["What is y if y=2*2-4+(3*2)"] 

    def multiple_layer_names(model_1:Any, model_2:Any) -> Tuple[List[str], List[str]]:
        layer_names_1 = make_layer_names(
            model_1,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        layer_names_2 = make_layer_names(
            model_2,
            block_step=block_step,
            include_input=include_input,
            force_include_output=force_include_output,
            include_subblocks=include_subblocks,
            decoder_layer_names=decoder_layer_names
        )

        return layer_names_1, layer_names_2
    
    model_1, tokenizer_1 = _load_model_tokenizer(model_1, tokenizer_1, model_precision_1)
    model_2, tokenizer_2 = _load_model_tokenizer(model_2, tokenizer_2, model_precision_2)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    if use_deterministic_backend:
        _set_deterministic_backend()

    layer_names_1, layer_names_2 = multiple_layer_names(model_1=model_1, model_2=model_2)

    make_lens_hooks(
        model_1, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_1, decoder_layer_names=decoder_layer_names, verbose=verbose
    )
    make_lens_hooks(
        model_2, start_ix=start_ix, end_ix=end_ix, layer_names=layer_names_2, decoder_layer_names=decoder_layer_names, verbose=verbose
    )

    input_ids_1 = text_to_input_ids(tokenizer_1, inputs, model_1, pad_to_max_length=pad_to_max_length)
    input_ids_1 = input_ids_1.to(next(model_1.parameters()).device)

    input_ids_2 = text_to_input_ids(tokenizer_2, inputs, model_2, pad_to_max_length=pad_to_max_length)
    input_ids_2 = input_ids_2.to(next(model_2.parameters()).device)

    seq_len = input_ids_1.shape[1]
    if end_ix > seq_len:
        print(f"Adjusting end_ix from {end_ix} to {seq_len}")
        end_ix = seq_len
    if start_ix >= seq_len or start_ix < 0:
        raise ValueError(f"start_ix {start_ix} is out of range for sequence length {seq_len}")
    if start_ix >= end_ix:
        raise ValueError(f"start_ix {start_ix} must be less than end_ix {end_ix}")

    # === Collect logits for both models ===
    layer_logits_1, layer_names_1 = collect_logits(model_1, input_ids_1, layer_names_1)
    layer_logits_2, _ = collect_logits(model_2, input_ids_2, layer_names_2)
    layer_logits_1 = safe_cast_logits(torch.tensor(layer_logits_1)).numpy()
    layer_logits_2 = safe_cast_logits(torch.tensor(layer_logits_2)).numpy()
    #layer_logits_1 = np.mean(layer_logits_1, axis=0, keepdims=True)  # shape becomes [1, seq_len, vocab_size]
    #layer_logits_2 = np.mean(layer_logits_2, axis=0, keepdims=True)

    layer_logits_1 = layer_logits_1[:, start_ix:end_ix, :]
    layer_logits_2 = layer_logits_2[:, start_ix:end_ix, :]

    if topk_mean:
        layer_preds_1, layer_probs_1, topk_scores_1 = postprocess_logits_topk(layer_logits_1, top_n=topk, return_scores=True)
        layer_preds_2, layer_probs_2, topk_scores_2 = postprocess_logits_topk(layer_logits_2, top_n=topk, return_scores=True)
    else:
        layer_preds_1, layer_probs_1 = postprocess_logits_topk(layer_logits_1, top_n=1, return_scores=False)
        layer_preds_2, layer_probs_2 = postprocess_logits_topk(layer_logits_2, top_n=1, return_scores=False)
        topk_scores_1 = None
        topk_scores_2 = None


    layer_probs_1 = np.nan_to_num(layer_probs_1, nan=1e-10, posinf=1.0, neginf=0.0)
    layer_probs_2 = np.nan_to_num(layer_probs_2, nan=1e-10, posinf=1.0, neginf=0.0)


    topk_indices_1 = np.argsort(layer_probs_1, axis=-1)[..., -topk:][..., ::-1]
    topk_scores_1 = np.take_along_axis(layer_probs_1, topk_indices_1, axis=-1)
    topk_indices_2 = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]
    topk_scores_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)

    if js:
        metric_type = 'js'
        map_color = 'Blues'
        title = f"JS ({'mean topk' if topk_mean else 'top-1'})"

        if topk_mean:
            topk_indices_2 = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]
            clipped_probs_1 = np.take_along_axis(layer_probs_1, topk_indices_2, axis=-1)
            clipped_probs_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)
            norm_probs_1 = clipped_probs_1 / np.clip(clipped_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            norm_probs_2 = clipped_probs_2 / np.clip(clipped_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            value_matrix = js_divergence(norm_probs_1, norm_probs_2)
        else:

            vocab_size = layer_probs_1.shape[-1]
            onehot_1 = top1_to_onehot(layer_preds_1, vocab_size)
            onehot_2 = top1_to_onehot(layer_preds_2, vocab_size)
            value_matrix = js_divergence(onehot_1, onehot_2)

    else:
        metric_type = 'nwd'
        map_color = 'Blues'
        title = f"NWD ({'mean topk' if topk_mean else 'top-1'})"

        if topk_mean:
            topk_indices_2 = np.argsort(layer_probs_2, axis=-1)[..., -topk:][..., ::-1]
            topk_probs_1 = np.take_along_axis(layer_probs_1, topk_indices_2, axis=-1)
            topk_probs_2 = np.take_along_axis(layer_probs_2, topk_indices_2, axis=-1)

            batch = layer_probs_1.shape[0]
            seq_len = layer_probs_1.shape[1]
            vocab_size = layer_probs_1.shape[-1]

            sparse_probs_1 = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
            sparse_probs_2 = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)

            for b in range(batch):
                for t in range(seq_len):
                    sparse_probs_1[b, t, topk_indices_2[b, t]] = topk_probs_1[b, t]
                    sparse_probs_2[b, t, topk_indices_2[b, t]] = topk_probs_2[b, t]

            sparse_probs_1 /= np.clip(sparse_probs_1.sum(axis=-1, keepdims=True), 1e-10, 1.0)
            sparse_probs_2 /= np.clip(sparse_probs_2.sum(axis=-1, keepdims=True), 1e-10, 1.0)

            value_matrix = nwd(sparse_probs_1, sparse_probs_2)

        else:
            vocab_size = layer_probs_1.shape[-1]
            onehot_1 = top1_to_onehot(layer_preds_1, vocab_size)
            onehot_2 = top1_to_onehot(layer_preds_2, vocab_size)
            value_matrix = nwd(onehot_1, onehot_2)
            value_matrix = value_matrix.sum(axis=-1)

    # Remove batch dim if batch size = 1
    if layer_logits_2.shape[0] == 1:
        layer_logits_2 = layer_logits_2.squeeze(0)

    # Fix dimension issue for topk=1
    if value_matrix.ndim == 1:
        value_matrix = np.expand_dims(value_matrix, axis=1)
        #value_matrix = np.repeat(value_matrix, layer_logits_1.shape[1], axis=1)
        value_matrix = np.repeat(value_matrix, layer_logits_2.shape[1], axis=1)

    # Plot using model 1’s annotations and divergence matrix
    fig = _topk_comparing_lens_fig(
        layer_logits=layer_logits_2,  # Use model 2's logits
        layer_preds=layer_preds_2,    # Use model 2's predictions
        layer_probs=layer_probs_2,   # Use model 2's probabilities
        topk_scores=topk_scores_2,   # Use model 2's top-k scores
        topk_indices=topk_indices_2, # Use model 2's top-k indices
        tokenizer=tokenizer_2,       # Use model 2's tokenizer
        input_ids=input_ids_2,       # Use model 2's input IDs
        start_ix=start_ix,
        layer_names=layer_names_2,   # Use model 2's layer names
        top_k=topk,
        normalize=True,
        metric_type=metric_type,
        map_color=map_color,
        value_matrix=value_matrix,
        title=title,
        block_step=block_step,
        token_font_size=token_font_size,
        label_font_size=label_font_size
    )

    clear_cuda_cache()
    return fig