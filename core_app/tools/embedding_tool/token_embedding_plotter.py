from typing import Tuple, List, Dict, Any, Union, Optional
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import lru_cache


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

def text_to_input_ids(tokenizer:Any, text:Union[str, List[str]], model:Optional[torch.nn.Module]=None, add_special_tokens:bool=True, pad_to_max_length=False) -> torch.Tensor:
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



def visualize_token_embedding_arithmetic(model, tokenizer, expression: str, top_n: int = 5):
    device = model.device
    base_emb = model.get_input_embeddings()
    token_list = expression.split()

    # Build the vector from the expression
    result_vector = None
    components = []
    for token in token_list:
        sign = 1
        if token.startswith('-'):
            sign = -1
            token = token[1:]
        elif token.startswith('+'):
            token = token[1:]
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            continue
        emb = base_emb(torch.tensor([token_id], device=device)).detach().cpu().numpy().squeeze()
        result_vector = emb * sign if result_vector is None else result_vector + emb * sign
        components.append((token, emb))

    # Get entire embedding matrix
    embedding_matrix = base_emb.weight.detach().cpu().numpy()  # [V, D]
    sim_scores = cosine_similarity(result_vector.reshape(1, -1), embedding_matrix)[0]
    top_indices = np.argsort(sim_scores)[-top_n:][::-1]

    top_tokens = [tokenizer.convert_ids_to_tokens([i])[0] for i in top_indices]
    top_vectors = embedding_matrix[top_indices]
    top_similarities = sim_scores[top_indices]

    # Stack all vectors: input tokens + result + top neighbors
    all_labels = [t for t, _ in components] + ['[RESULT]'] + top_tokens
    all_vectors = [v for _, v in components] + [result_vector] + list(top_vectors)

    # Project to 2D
    coords = PCA(n_components=2).fit_transform(np.vstack(all_vectors))

    fig = go.Figure()

    # Plot input tokens
    for i, (token, coord) in enumerate(zip(all_labels, coords)):
        text = token
        if token in top_tokens:
            sim = top_similarities[top_tokens.index(token)]
            text += f" ({sim:.2f})"
        fig.add_trace(go.Scatter(
            x=[coord[0]], y=[coord[1]],
            mode="markers+text",
            marker=dict(size=12 if token != '[RESULT]' else 16, symbol="circle"),
            text=[text], name=token,
            textposition="top center"
        ))

    # Add arrows from inputs to result
    for i in range(len(components)):
        from_pt = coords[i]
        to_pt = coords[len(components)]
        fig.add_trace(go.Scatter(
            x=[from_pt[0], to_pt[0]],
            y=[from_pt[1], to_pt[1]],
            mode='lines',
            line=dict(width=1, color='gray'),
            showlegend=False
        ))

    fig.update_layout(
        title="Token Embedding Arithmetic + Nearest Neighbors",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        height=650,
        #template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        #paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig
