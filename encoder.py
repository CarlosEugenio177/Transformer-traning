import torch
from attention import add_and_norm, FeedForward, SelfAttention


def run_encoder(X, num_layers=6):
    X_entrada = X.clone()
    X_atual = X.clone()

    attention_layers = [SelfAttention(X.shape[-1]) for _ in range(num_layers)]
    ffn_layers = [FeedForward(X.shape[-1]) for _ in range(num_layers)]

    for layer in range(num_layers):
        X_att, _ = attention_layers[layer](X_atual)
        X_norm1 = add_and_norm(X_atual, X_att)
        X_ffn = ffn_layers[layer](X_norm1)
        X_out = add_and_norm(X_norm1, X_ffn)
        X_atual = X_out

    if X_atual.shape == X_entrada.shape:
        print(f"Dimensões de X mantidas: {X_atual.shape}")
    else:
        raise ValueError("Erro: Dimensões de X foram alteradas")

    valores_alterados = not torch.allclose(X_atual, X_entrada)
    assert valores_alterados, "Erro: Os valores de X não foram alterados"

    print("\nRepresentações contextualizadas geradas: Vetor Z obtido após o processamento pelo Encoder")
    print("\nVALIDAÇÃO DE SANIDADE: PASSOU EM TODAS AS VERIFICAÇÕES")

    return X_atual


if __name__ == "__main__":
    X = torch.rand(1, 10, 64)
    Z = run_encoder(X)