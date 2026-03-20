import numpy as np
from data import X
from attention import self_attention, add_and_norm, feed_forward

X_entrada = X.copy()
X_atual = X.copy()

for layer in range(1, 7):
    X_att = self_attention(X_atual)
    X_Norm1 = add_and_norm(X_atual, X_att)
    X_ffn = feed_forward(X_Norm1)
    X_out = add_and_norm(X_Norm1, X_ffn)
    X_atual = X_out

if X_atual.shape == X_entrada.shape and X_atual.shape[-1] == 64:

    print(f'Dimensões de X mantidas: {X_atual.shape}')

else:

    raise ValueError(f'Erro: Dimensões de X foram alteradas')

valores_alterados = not np.allclose(X_atual, X_entrada)
assert valores_alterados, 'Erro: Os valores de X não foram alterados'

print('\nRepresentações contextualizadas geradas: Vetor Z obtido após o processamento pelo Encoder')
print('\nVALIDAÇÃO DE SANIDADE: PASSOU EM TODAS AS VERIFICAÇÕES')

Z = X_atual