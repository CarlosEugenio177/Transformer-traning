import torch
import torch.nn as nn
from attention import softmax, scaled_dot_product_attention


def look_ahead_mask(seq_len, device=None):
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(), float("-inf"))
    mask = mask.unsqueeze(0)
    return mask


class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, encoder_out, decoder_state):
        Q = decoder_state @ self.W_Q
        K = encoder_out @ self.W_K
        V = encoder_out @ self.W_V

        scores = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(encoder_out.shape[-1], dtype=encoder_out.dtype, device=encoder_out.device))

        attention = softmax(scores)
        output = attention @ V

        return output


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, Y):
        Q = Y @ self.W_Q
        K = Y @ self.W_K
        V = Y @ self.W_V

        mask = look_ahead_mask(Y.shape[1], device=Y.device)

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return output, attention_weights


def generate_next_token(current_sequence, encoder_out, vocab_size=10000):
    probs = torch.rand(vocab_size, device=encoder_out.device)
    probs = probs / probs.sum()
    return probs


def autoregressive_loop():
    encoder_out = torch.rand(1, 10, 512)

    vocab = ["No", "i", "am", "your", "father", ".", "<EOS>"]
    sequence = ["<START>"]

    while True:
        probs = generate_next_token(sequence, encoder_out)

        token_index = torch.argmax(probs).item()
        next_token = vocab[token_index % len(vocab)]

        sequence.append(next_token)

        if next_token == "<EOS>":
            break

    print("Frase gerada:")
    print(" ".join(sequence))


if __name__ == "__main__":
    print("Teste máscara causal:")
    print(look_ahead_mask(5)[0])

    print("\nTeste cross-attention:")
    encoder_output = torch.rand(1, 10, 512)
    decoder_state = torch.rand(1, 4, 512)

    cross_attention = CrossAttention(d_model=512)
    cross_output = cross_attention(encoder_output, decoder_state)
    print("Cross-attention shape:", cross_output.shape)

    print("\nTeste masked self-attention:")
    Y = torch.rand(1, 4, 512)
    masked_attention = MaskedSelfAttention(d_model=512)
    masked_output, masked_weights = masked_attention(Y)
    print("Masked self-attention output shape:", masked_output.shape)
    print("Masked self-attention weights shape:", masked_weights.shape)

    print("\nTeste loop auto-regressivo:")
    autoregressive_loop()