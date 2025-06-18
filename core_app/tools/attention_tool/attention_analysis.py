import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import lru_cache


def clear_cuda_cache():
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()

# cache so we don’t re-load on every callback
@lru_cache(maxsize=2)
def load_model_tokenizer(model_id:str, tok_id:str, quant_config:str|None):
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
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            return_dict=True,
            output_hidden_states=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True
        )

    return model, tok

def get_attention_figure(model, tokenizer, text, layer_idx, head_idx, apply_softmax=True):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions  # list of [batch, heads, q_len, k_len]
    attn = attentions[layer_idx][0, head_idx]  # [q_len, k_len]

    if apply_softmax:
        attn = torch.nn.functional.softmax(attn, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    fig = go.Figure(data=go.Heatmap(
        z=attn.numpy(),
        x=tokens,
        y=tokens,
        colorscale="Viridis"
    ))

    fig.update_layout(
        title=f"Attention Head {head_idx} @ Layer {layer_idx}",
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        margin=dict(l=60, r=60, t=40, b=60),
        width=max(700, 40 * len(tokens)),
        height=max(600, 40 * len(tokens))
    )
    
    return fig


def get_attention_weights(model, tokenizer, text, layer=0, head=0):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[layer][0, head]  # Shape: [seq_len, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return attn.cpu(), tokens


def plot_attention_heatmap(attn, tokens, top_k=None):
    attn_np = attn.numpy()
    if top_k:
        # Show only top_k attention per row
        for i in range(attn_np.shape[0]):
            top_indices = np.argsort(attn_np[i])[-top_k:]
            mask = np.ones_like(attn_np[i])
            mask[top_indices] = 0
            attn_np[i][mask == 1] = 0.0

    fig = go.Figure(data=go.Heatmap(
        z=attn_np,
        x=tokens,
        y=tokens,
        colorscale="Viridis",
        colorbar=dict(title="Attention")
    ))
    fig.update_layout(
        title="Attention Heatmap",
        xaxis_title="Key",
        yaxis_title="Query",
        autosize=True,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def compute_attention_entropy(attn):
    probs = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
    entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)
    return entropy.cpu().numpy()


def plot_entropy_heatmap(entropy, tokens):
    fig = go.Figure(data=go.Bar(
        x=tokens,
        y=entropy,
        marker=dict(color=entropy, colorscale="YlGnBu"),
    ))
    fig.update_layout(
        title="Attention Entropy per Token",
        xaxis_title="Token",
        yaxis_title="Entropy (bits)",
        height=400,
        margin=dict(t=40, b=40),
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig


def plot_qk_embeddings(model, tokenizer, text, layer=0, mode="pca"):
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model.model(**inputs, output_hidden_states=True)

    q_proj = model.model.layers[layer].self_attn.q_proj
    k_proj = model.model.layers[layer].self_attn.k_proj
    hidden = outputs.hidden_states[layer][0]

    Q = q_proj(hidden).detach().numpy()
    K = k_proj(hidden).detach().numpy()

    reducer = PCA(n_components=2) if mode == "pca" else TSNE(n_components=2)
    q_emb = reducer.fit_transform(Q)
    k_emb = reducer.fit_transform(K)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_emb[:, 0], y=q_emb[:, 1], mode="markers+text", name="Q",
                             text=tokens, textposition="top center", marker=dict(color="red")))
    fig.add_trace(go.Scatter(x=k_emb[:, 0], y=k_emb[:, 1], mode="markers+text", name="K",
                             text=tokens, textposition="bottom center", marker=dict(color="blue")))

    fig.update_layout(title=f"Q/K Embeddings ({mode.upper()})", height=500, plot_bgcolor='rgba(0, 0, 0, 0)')
    return fig