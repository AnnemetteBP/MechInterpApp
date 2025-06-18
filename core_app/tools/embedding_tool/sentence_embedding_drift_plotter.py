import torch
import numpy as np
from sklearn.decomposition import PCA
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

def get_sentence_embedding(model, tokenizer, sentence, layer=-1, method="mean"):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer].squeeze(0)  # [seq_len, hidden]

    if method == "mean":
        return hidden.mean(dim=0).cpu().numpy()
    elif method == "cls":
        return hidden[0].cpu().numpy()
    else:
        raise ValueError("Invalid pooling method")


def plot_sentence_embedding_drift(sentences, model_fp, tokenizer_fp, model_q, tokenizer_q, top_layer=-1, method="mean"):
    vectors_fp = [get_sentence_embedding(model_fp, tokenizer_fp, s, top_layer, method) for s in sentences]
    vectors_q = [get_sentence_embedding(model_q, tokenizer_q, s, top_layer, method) for s in sentences]

    labels = []
    all_vecs = []

    for i, sentence in enumerate(sentences):
        all_vecs.append(vectors_fp[i])
        labels.append(f"FP: {sentence[:30]}")
        all_vecs.append(vectors_q[i])
        labels.append(f"Q: {sentence[:30]}")

    coords = PCA(n_components=2).fit_transform(np.vstack(all_vecs))
    fig = go.Figure()

    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[coords[i][0]], y=[coords[i][1]],
            mode="markers+text",
            text=[label],
            name=label,
            textposition="top center",
            marker=dict(size=10)
        ))

    fig.update_layout(
        title="Sentence Embedding Drift (FP vs Quantized)",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        height=700,
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig
