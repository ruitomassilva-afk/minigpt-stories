import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import gradio as gr
from safetensors.numpy import load_file

# -----------------------------
# Configuração
# -----------------------------
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab

MAXLEN = 128
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6

WEIGHTS_PATH = "minigpt_weights.safetensors"

# -----------------------------
# Arquitetura do Modelo
# -----------------------------
class Embedding(nnx.Module):
    def __init__(self, vocab_size, d_model, max_len, rngs):
        self.token_emb = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, d_model, rngs=rngs)

    def __call__(self, x):
        _, t = x.shape
        pos = jnp.arange(t)[None, :]
        return self.token_emb(x) + self.pos_emb(pos)

class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(d_model, 4 * d_model, rngs=rngs),
            nnx.gelu,
            nnx.Linear(4 * d_model, d_model, rngs=rngs),
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nnx.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, rngs):
        self.embedding = Embedding(vocab_size, d_model, max_len, rngs)
        self.transformer_blocks = [TransformerBlock(d_model, n_heads, rngs) for _ in range(n_layers)]
        self.output_layer = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.output_layer(x)

# -----------------------------
# Carregamento de Pesos
# -----------------------------
def load_weights(model, path):
    if not os.path.exists(path):
        print(f"⚠️ Ficheiro {path} não encontrado.")
        return
    
    tensors = load_file(path)
    state = nnx.state(model)
    flat = state.flat_state()
    
    count = 0
    for key_tuple, var in flat.items():
        # Criar a chave string (ex: "embedding.token_emb.embedding.value")
        key_str = ".".join(str(k) for k in key_tuple)
        if key_str in tensors:
            var.value = jnp.array(tensors[key_str])
            count += 1
            
    print(f"✅ Carregados {count}/{len(flat)} tensores com sucesso.")

# Inicializar modelo e carregar pesos
rngs = nnx.Rngs(0)
model = MiniGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAXLEN, rngs)
load_weights(model, WEIGHTS_PATH)

# -----------------------------
# Lógica de Geração
# -----------------------------
def generate(prompt, max_new_tokens, temperature, top_k):
    if not prompt.strip(): return "Escreve algo para começar..."
    
    input_ids = jnp.array([tokenizer.encode(prompt)], dtype=jnp.int32)
    
    for _ in range(int(max_new_tokens)):
        curr = input_ids[:, -MAXLEN:]
        logits = model(curr)[:, -1, :] / float(temperature)
        
        # Top-K sampling
        top_vals, _ = jax.lax.top_k(logits, int(top_k))
        logits = jnp.where(logits < top_vals[:, -1:], -jnp.inf, logits)
        
        key = jax.random.PRNGKey(int(time.time_ns() % (2**31 - 1)))
        next_tok = jax.random.categorical(key, logits).astype(jnp.int32)[:, None]
        input_ids = jnp.concatenate([input_ids, next_tok], axis=1)
        
        if int(next_tok[0, 0]) == tokenizer.eot_token: break
        
    return tokenizer.decode(np.array(input_ids[0]).tolist())

# -----------------------------
# Interface Gradio
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Minigpt Stories 🌖\nModelo pedagógico treinado de raiz.")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Prompt (Inglês)", value="Once upon a time")
            tokens = gr.Slider(10, 200, value=50, label="Tokens")
            temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperatura")
            k = gr.Slider(1, 100, value=40, label="Top-K")
            btn = gr.Button("Gerar História", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Resultado", lines=10)
            
    btn.click(generate, [txt, tokens, temp, k], out)

demo.launch()
