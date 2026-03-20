from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

MAX_SAMPLES = 50
MAX_LENGTH = 32
BATCH_SIZE = 8

dataset = load_dataset("bentrevett/multi30k", split="train")
dataset = dataset.select(range(MAX_SAMPLES))

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

START_TOKEN = "[CLS]"
END_TOKEN = "[SEP]"
PAD_TOKEN_ID = tokenizer.pad_token_id


def process_example(example):
    src = example["en"]
    tgt = example["de"]

    tgt = START_TOKEN + " " + tgt + " " + END_TOKEN

    src_tokens = tokenizer(
        src,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    tgt_tokens = tokenizer(
        tgt,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )

    return {
        "encoder_input": src_tokens["input_ids"],
        "decoder_full": tgt_tokens["input_ids"],
    }


processed_dataset = dataset.map(process_example)

processed_dataset = processed_dataset.remove_columns(
    [col for col in processed_dataset.column_names if col not in ["encoder_input", "decoder_full"]]
)


def get_tensors():
    encoder_inputs = torch.tensor(processed_dataset["encoder_input"], dtype=torch.long)
    decoder_full = torch.tensor(processed_dataset["decoder_full"], dtype=torch.long)

    decoder_inputs = decoder_full[:, :-1]
    target_outputs = decoder_full[:, 1:]

    return encoder_inputs, decoder_inputs, target_outputs


def get_dataloader(batch_size=BATCH_SIZE):
    encoder_inputs, decoder_inputs, target_outputs = get_tensors()

    dataset = TensorDataset(encoder_inputs, decoder_inputs, target_outputs)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    enc, dec_in, tgt_out = get_tensors()

    print(enc.shape)
    print(dec_in.shape)
    print(tgt_out.shape)

    batch = next(iter(get_dataloader()))
    print(batch[0].shape, batch[1].shape, batch[2].shape)