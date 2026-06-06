import sys
from pathlib import Path

import torch

from config import get_config, latest_weights_file_path
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from datasets import load_dataset
from tokenizers import Tokenizer


def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config        = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_src"]))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_tgt"]))))

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError("No trained weights found. Run train_wb.py first.")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    # If sentence is a digit, treat it as an index into the dataset
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds_raw = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="all")
        ds = BilingualDataset(ds_raw, tokenizer_src, tokenizer_tgt,
                              config["lang_src"], config["lang_tgt"], config["seq_len"])
        sentence = ds[idx]["src_text"]
        label    = ds[idx]["tgt_text"]

    seq_len = config["seq_len"]
    model.eval()
    with torch.no_grad():
        source_ids = tokenizer_src.encode(sentence).ids
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(source_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
            torch.tensor(
                [tokenizer_src.token_to_id("[PAD]")] * (seq_len - len(source_ids) - 2),
                dtype=torch.int64,
            ),
        ], dim=0).unsqueeze(0).to(device)   # (1, seq_len)

        source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).int().to(device)
        encoder_output = model.encode(source, source_mask)

        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id("[SOS]")).type_as(source).to(device)

        if label:
            print(f"{'SOURCE:':>12} {sentence}")
            print(f"{'TARGET:':>12} {label}")
        else:
            print(f"{'SOURCE:':>12} {sentence}")
        print(f"{'PREDICTED:':>12}", end=" ")

        while decoder_input.size(1) < seq_len:
            decoder_mask = (
                torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1)
                .type(torch.int)
                .type_as(source_mask)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ], dim=1)
            print(tokenizer_tgt.decode([next_word.item()]), end=" ")
            if next_word == tokenizer_tgt.token_to_id("[EOS]"):
                break

        print()

    return tokenizer_tgt.decode(decoder_input[0].tolist())


if __name__ == "__main__":
    sentence = sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student."
    translate(sentence)
