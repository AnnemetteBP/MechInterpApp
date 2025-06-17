from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any

import os, json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import numpy as np
import plotly.graph_objects as go

from functools import lru_cache

from .sae_class import SAE
from .sae_hooks import get_layer_activations


def clear_cuda_cache():
    """Clear GPU cache to avoid memory errors during operations"""
    torch.cuda.empty_cache()

def clean_token(token: str) -> str:
    if token.startswith("Ġ"):
        return " " + token[1:]
    elif token.startswith("Ċ"):
        return "\\n" + token[1:]
    else:
        return token

def _plot_token_heatmap(
    tokens:List[str],
    feature_token_matrix:torch.Tensor,
    top_k:int=5,
    saliency:torch.Tensor|None=None,
    tokens_per_row:int=12,
) -> go.Figure:
    
    num_tokens = len(tokens)
    token_feature_matrix = feature_token_matrix.T  # [num_tokens x top_k_features]

    num_rows = math.ceil(num_tokens / tokens_per_row)
    z = np.zeros((num_rows, tokens_per_row))
    text = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]
    hovertext = [["" for _ in range(tokens_per_row)] for _ in range(num_rows)]

    # Normalize saliency
    if saliency is not None:
        norm_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-5)
        norm_saliency = norm_saliency.numpy()

    for idx, token in enumerate(tokens):
        row = idx // tokens_per_row
        col = idx % tokens_per_row

        clean_tok = clean_token(token)
        token_scores = token_feature_matrix[idx]
        top_vals, top_idx = token_scores.topk(top_k)

        z[row][col] = norm_saliency[idx] if saliency is not None else top_vals[0].item()
        text[row][col] = clean_tok

        #hover = f"Token: {clean_tok}\nTop-{top_k} Features:\n"
        #for f_idx, f_val in zip(top_idx, top_vals):
            #val = f_val.item()
            #norm_val = (val - top_vals.min().item()) / (top_vals.max().item() - top_vals.min().item() + 1e-5)
            #saliency_marker = "●" * int(norm_val * 5)  # Simulated saliency
            #hover += f"  {saliency_marker} Feature {f_idx.item()}: {val:.3f}\n"
        #hovertext[row][col] = hover

        hover = f"<b>Token:</b> {clean_tok}<br><b>Saliency:</b> {norm_saliency[idx]:.3f}<br><br>"
        hover += f"<b>Top-{top_k} Tokens:</b><br>"
        for rank, (f_idx, f_val) in enumerate(zip(top_idx, top_vals), 1):
            tok = clean_token(tokens[f_idx.item()]) if f_idx.item() < len(tokens) else f"F{f_idx.item()}"
            norm_val = (f_val.item() - top_vals.min().item()) / (top_vals.max().item() - top_vals.min().item() + 1e-5)
            bg_color = f"rgba({255*(1-norm_val):.0f},{255*norm_val:.0f},150,0.6)"
            hover += f"<span style='background-color:{bg_color};padding:2px;'>Top-{rank}: {tok} ({f_val.item():.2f})</span><br>"

        hovertext[row][col] = hover

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        hoverinfo="text",
        hovertext=hovertext,
        colorscale='reds_r',  
        reversescale=True,
        showscale=True,
        texttemplate="%{text}",
        #textfont={"size": 10, "color": "black", "family": "Times New Roman"},
        textfont={"size": 10, "family": "DejaVu Sans"},
        xgap=2, ygap=2
    ))

    fig.update_layout(
        title=f"Top-k {top_k} Token Activation",
        title_font=dict(size=12, family="DejaVu Sans"),
        width=max(600, tokens_per_row * 50),
        height=num_rows * 50 + 100,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showticklabels=False),
        #yaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange='reversed'),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return fig


def _run_multi_layer_sae(
        model:Any,
        tokenizer:Any,
        text:str,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
        model_to_eval:bool=True,
        deterministic_sae:bool=True,   
) -> go.Figure:

    if model_to_eval:
        model.eval()
    
    all_layer_outputs = {}

    for layer_idx in target_layers:
        print(f"\nRunning SAE on layer {layer_idx}")
        token_ids, hidden = get_layer_activations(model, tokenizer, text, target_layer_idx=layer_idx)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        sae = SAE(input_dim=hidden.shape[1], dict_size=512, sparsity_lambda=1e-3, deterministic_sae=deterministic_sae)
        sae.train_sae(hidden, epochs=10, batch_size=4)

        codes = sae.encode(hidden).detach()
        normed = F.normalize(codes, dim=1)
        saliency = normed.abs().max(dim=1).values

        layer_data = {
            'tokens': tokens,
            'saliency': saliency,
            'codes': codes,
            'hidden': hidden,
            'dict': sae.decoder.weight.detach()
        }

        all_layer_outputs[f"layer_{layer_idx}"] = layer_data
        #feature_token_matrix = codes.abs().T
        # Rank neurons by total activation
        total_per_feature = codes.abs().sum(dim=0)  # shape: [dict_size]
        topk_indices = total_per_feature.topk(top_k).indices
        feature_token_matrix = codes[:, topk_indices].abs().T  # shape: [top_k x num_tokens]

        fig = _plot_token_heatmap(
            tokens=tokens,
            feature_token_matrix=feature_token_matrix,
            top_k=top_k,
            saliency=saliency,
            tokens_per_row=tokens_per_row
        )

    return fig


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

def plot_sae_heatmap(
        model_path:Any,
        tokenizer_path:Any,
        inputs:Any,
        model_precision:Optional[str|None]=None,
        top_k:int=5,
        tokens_per_row:int=12,
        target_layers:List[int]=[5,10,15],
        model_to_eval:bool=True,
        deterministic_sae:bool=True,
        token_font_size:int=12,
        label_font_size:int=20,
) -> None:
    """
    Run multi-layer SAE analysis
    eval() to disable dropout and uses running statistics for batch norm instead of batch-wise statistics.
    This ensures the model behaves consistently during evaluation or inference (e.g., stable activations for SAE), not training.
    """
    model, tokenizer = _load_model_tokenizer(model_path, tokenizer_path, model_precision)
    
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    fig = _run_multi_layer_sae(
        model=model,
        tokenizer=tokenizer,
        text=inputs,
        top_k=top_k,
        tokens_per_row=tokens_per_row,
        target_layers=target_layers,
        model_to_eval=model_to_eval,
        deterministic_sae=deterministic_sae,
    )

    clear_cuda_cache()
    return fig