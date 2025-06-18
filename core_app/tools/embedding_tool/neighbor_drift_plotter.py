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

def compute_vector_from_expression(expression: str, tokenizer, model):
    tokens = expression.split()
    embedding = model.get_input_embeddings()
    device = model.device
    vec = None
    components = []

    for token in tokens:
        sign = 1
        if token.startswith('-'):
            sign = -1
            token = token[1:]
        elif token.startswith('+'):
            token = token[1:]
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            continue
        emb = embedding(torch.tensor([token_id], device=device)).detach().cpu().numpy().squeeze()
        vec = emb * sign if vec is None else vec + emb * sign
        components.append((token, emb))
    return vec, components


def get_top_neighbors(vec, tokenizer, embedding, top_n=5):
    all_embeddings = embedding.weight.detach().cpu().numpy()
    sims = cosine_similarity(vec.reshape(1, -1), all_embeddings)[0]
    indices = np.argsort(sims)[-top_n:][::-1]
    tokens = [tokenizer.convert_ids_to_tokens([i])[0] for i in indices]
    vectors = all_embeddings[indices]
    scores = sims[indices]
    return list(zip(tokens, vectors, scores))


def plot_fp_vs_quantized_neighbors(expression, model_fp, tokenizer_fp, model_q, tokenizer_q, top_n=5):
    result_fp, components_fp = compute_vector_from_expression(expression, tokenizer_fp, model_fp)
    result_q, components_q = compute_vector_from_expression(expression, tokenizer_q, model_q)

    neighbors_fp = get_top_neighbors(result_fp, tokenizer_fp, model_fp.get_input_embeddings(), top_n)
    neighbors_q = get_top_neighbors(result_q, tokenizer_q, model_q.get_input_embeddings(), top_n)

    all_labels = [f"FP:{t}" for t, _ in components_fp] + ["[RESULT_FP]"] + [f"FP→{t}" for t, _, _ in neighbors_fp] + \
                 [f"Q:{t}" for t, _ in components_q] + ["[RESULT_Q]"] + [f"Q→{t}" for t, _, _ in neighbors_q]

    all_vectors = [v for _, v in components_fp] + [result_fp] + [v for _, v, _ in neighbors_fp] + \
                  [v for _, v in components_q] + [result_q] + [v for _, v, _ in neighbors_q]

    coords = PCA(n_components=2).fit_transform(np.vstack(all_vectors))
    fig = go.Figure()

    for label, coord in zip(all_labels, coords):
        fig.add_trace(go.Scatter(
            x=[coord[0]], y=[coord[1]],
            mode="markers+text",
            text=[label],
            name=label,
            textposition="top center",
            marker=dict(size=10)
        ))

    fig.update_layout(
        title="Neighbor Drift: FP vs Quantized Embedding Result",
        height=700,
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig