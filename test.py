import torch
import torch.nn as nn
import torch.optim as optim

from data import dataset, tokenizer, PAD_TOKEN_ID
from transformer import Transformer
from train import TokenEmbedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
EPOCHS = 30
MAX_LENGTH = 32

VOCAB_SIZE = tokenizer.vocab_size
START_TOKEN_ID = tokenizer.cls_token_id
END_TOKEN_ID = tokenizer.sep_token_id


def prepare_single_example(example_index=0):
    example = dataset[example_index]

    src_text = example["en"]
    tgt_text = example["de"]

    encoder_tokens = tokenizer(
        src_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )["input_ids"]

    target_text = "[CLS] " + tgt_text + " [SEP]"

    decoder_full = tokenizer(
        target_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"]

    decoder_input_ids = decoder_full[:, :-1]
    target_output_ids = decoder_full[:, 1:]

    return src_text, tgt_text, encoder_tokens, decoder_input_ids, target_output_ids


def train_single_example():
    src_text, tgt_text, encoder_input_ids, decoder_input_ids, target_output_ids = prepare_single_example(0)

    encoder_input_ids = encoder_input_ids.to(DEVICE)
    decoder_input_ids = decoder_input_ids.to(DEVICE)
    target_output_ids = target_output_ids.to(DEVICE)

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

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

    return embedding, model, src_text, tgt_text


def greedy_decode(embedding, model, src_text):
    model.eval()
    embedding.eval()

    encoder_input_ids = tokenizer(
        src_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )["input_ids"].to(DEVICE)

    with torch.no_grad():
        encoder_embeddings = embedding(encoder_input_ids)
        Z = model.encoder(encoder_embeddings)

        generated_ids = [START_TOKEN_ID]

        for _ in range(MAX_LENGTH - 1):
            decoder_input_ids = torch.tensor([generated_ids], dtype=torch.long, device=DEVICE)
            decoder_embeddings = embedding(decoder_input_ids)

            Y = decoder_embeddings
            for layer in model.decoder_layers:
                Y = layer(Y, Z)

            logits = model.output_projection(Y)
            next_token_id = torch.argmax(logits[0, -1]).item()

            generated_ids.append(next_token_id)

            if next_token_id == END_TOKEN_ID:
                break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, generated_text


if __name__ == "__main__":
    embedding, model, src_text, tgt_text = train_single_example()

    generated_ids, generated_text = greedy_decode(embedding, model, src_text)

    print("\nFrase de entrada:")
    print(src_text)

    print("\nTradução esperada:")
    print(tgt_text)

    print("\nTradução gerada:")
    print(generated_text)

    print("\nIDs gerados:")
    print(generated_ids)