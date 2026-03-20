from datasets import load_dataset
from transformers import AutoTokenizer
import torch


MAX_SAMPLES = 500        
MAX_LENGTH = 32         


dataset = load_dataset("bentrevett/multi30k", split="train")


dataset = dataset.select(range(MAX_SAMPLES))


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

START_TOKEN = "[CLS]"
END_TOKEN = "[SEP]"
PAD_TOKEN_ID = tokenizer.pad_token_id


def process_example(example):
    src = example["en"]  # inglês
    tgt = example["de"]  # alemão

    tgt = START_TOKEN + " " + tgt + " " + END_TOKEN

    src_tokens = tokenizer(
        src,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    tgt_tokens = tokenizer(
        tgt,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return {
        "encoder_input": src_tokens["input_ids"].squeeze(0),
        "decoder_input": tgt_tokens["input_ids"].squeeze(0)
    }


processed_dataset = dataset.map(process_example)


def get_tensors():
    encoder_inputs = torch.stack(processed_dataset["encoder_input"])
    decoder_inputs = torch.stack(processed_dataset["decoder_input"])

    return encoder_inputs, decoder_inputs


if __name__ == "__main__":
    enc, dec = get_tensors()

    print("Encoder shape:", enc.shape)
    print("Decoder shape:", dec.shape)

    print("\nExemplo:")
    print("Encoder:", enc[0])
    print("Decoder:", dec[0])