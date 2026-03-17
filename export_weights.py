import numpy as np
from safetensors.numpy import save_file
from flax import nnx

def export_model_to_safetensors(model: nnx.Module, filename: str = "minigpt_weights.safetensors"):
    """
    Exporta os pesos de um modelo Flax NNX para o formato Safetensors.
    
    Args:
        model: O modelo instanciado (MiniGPT).
        filename: Nome do ficheiro de saída.
    """
    print(f"📦 A iniciar exportação para {filename}...")
    
    # Extrai o estado do modelo (pesos)
    _, state = nnx.split(model)
    flat = state.flat_state()

    weights = {}
    for key_tuple, var in flat.items():
        # Converte a tupla de chaves (ex: ('embedding', 'token_emb', 'value')) 
        # numa string separada por pontos para compatibilidade com Safetensors.
        key_str = ".".join(str(k) for k in key_tuple)
        
        # Garante que o valor é um array numpy
        weights[key_str] = np.array(var.value)

    # Guarda o ficheiro
    save_file(weights, filename)
    
    print(f"✅ Exportação concluída com sucesso!")
    print(f"Total de tensores exportados: {len(weights)}")

if __name__ == "__main__":
    print("Este script deve ser importado no teu notebook de treino para exportar os pesos finais.")
    print("Exemplo de uso no Colab:")
    print(">> from export_weights import export_model_to_safetensors")
    print(">> export_model_to_safetensors(model, 'minigpt_weights.safetensors')")
