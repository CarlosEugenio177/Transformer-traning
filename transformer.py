import torch
import torch.nn as nn
from attention import add_and_norm, FeedForward, SelfAttention, softmax
from decoder import look_ahead_mask, CrossAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attention = SelfAttention(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, X):
        X_att, _ = self.self_attention(X)
        X_norm1 = add_and_norm(X, X_att)

        X_ffn = self.feed_forward(X_norm1)
        X_out = add_and_norm(X_norm1, X_ffn)

        return X_out


class EncoderStack(nn.Module):
    def __init__(self, d_model, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model) for _ in range(num_layers)])

    def forward(self, X):
        Z = X

        for layer in self.layers:
            Z = layer(Z)

        return Z


class DecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attention = SelfAttention(d_model)
        self.cross_attention = CrossAttention(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, Y, Z):
        mask = look_ahead_mask(Y.shape[1], device=Y.device)

        Y_att, _ = self.self_attention(Y, mask)
        Y_norm1 = add_and_norm(Y, Y_att)

        Y_cross = self.cross_attention(Z, Y_norm1)
        Y_norm2 = add_and_norm(Y_norm1, Y_cross)

        Y_ffn = self.feed_forward(Y_norm2)
        Y_out = add_and_norm(Y_norm2, Y_ffn)

        return Y_out


class OutputProjection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, vocab_size) * 0.01)

    def forward(self, Y):
        logits = Y @ self.W
        return logits


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.encoder = EncoderStack(d_model=d_model, num_layers=num_layers)
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model) for _ in range(num_layers)])
        self.output_projection = OutputProjection(d_model, vocab_size)

    def forward(self, encoder_input, decoder_input):
        Z = self.encoder(encoder_input)

        Y = decoder_input
        for layer in self.decoder_layers:
            Y = layer(Y, Z)

        logits = self.output_projection(Y)
        return logits


def run_inference():
    vocab = ["<START>", "Thinking", "Machines", "No", "i", "am", "your", "father", ".", "<EOS>"]

    model = Transformer(vocab_size=len(vocab), d_model=64, num_layers=2)

    encoder_input = torch.rand(1, 2, 64)
    Z = model.encoder(encoder_input)

    sequence = ["<START>"]
    Y = torch.rand(1, 1, 64)

    while True:
        for layer in model.decoder_layers:
            Y = layer(Y, Z)

        logits = model.output_projection(Y)
        probs = softmax(logits)

        token_index = torch.argmax(probs[0, -1]).item()
        next_token = vocab[token_index]

        sequence.append(next_token)

        if next_token == "<EOS>":
            break

        new_vec = torch.rand(1, 1, 64)
        Y = torch.cat([Y, new_vec], dim=1)

        if len(sequence) > 20:
            break

    print("Entrada simulada do encoder: Thinking Machines")
    print("Sequência gerada:")
    print(" ".join(sequence))


if __name__ == "__main__":
    run_inference()