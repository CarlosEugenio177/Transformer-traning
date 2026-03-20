import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader, tokenizer, PAD_TOKEN_ID
from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 128
NUM_LAYERS = 2
EPOCHS = 10
LEARNING_RATE = 1e-3

VOCAB_SIZE = tokenizer.vocab_size


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)

    def forward(self, x):
        return self.embedding(x)


def train():
    dataloader = get_dataloader()

    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL).to(DEVICE)
    model = Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.Adam(
        list(embedding.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        embedding.train()
        model.train()

        epoch_loss = 0.0

        for encoder_input_ids, decoder_input_ids, target_output_ids in dataloader:
            encoder_input_ids = encoder_input_ids.to(DEVICE)
            decoder_input_ids = decoder_input_ids.to(DEVICE)
            target_output_ids = target_output_ids.to(DEVICE)

            encoder_embeddings = embedding(encoder_input_ids)
            decoder_embeddings = embedding(decoder_input_ids)

            logits = model(encoder_embeddings, decoder_embeddings)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target_output_ids.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    return embedding, model


if __name__ == "__main__":
    train()