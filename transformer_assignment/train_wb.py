import warnings
import os
import random
import io
import zipfile
import requests
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import torchmetrics

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """Generate a translation one token at a time, always picking the most
    probable next token (greedy search)."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Encode source once and reuse at every decoding step
    encoder_output = model.encode(source, source_mask)

    # Start with [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out  = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])                    # logits for the last position
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
        ], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt,
                   max_len, device, print_msg, global_step, num_examples=2):
    """Run greedy decoding on a handful of validation examples and log BLEU."""
    model.eval()
    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(validation_ds):
            if i == num_examples:
                break

            encoder_input = batch["encoder_input"].to(device)  # (1, seq_len)
            encoder_mask  = batch["encoder_mask"].to(device)

            model_out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, max_len, device,
            )
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(batch["src_text"][0])
            expected.append(batch["tgt_text"][0])
            predicted.append(model_out_text)

            print_msg("-" * 80)
            print_msg(f"SOURCE:    {batch['src_text'][0]}")
            print_msg(f"TARGET:    {batch['tgt_text'][0]}")
            print_msg(f"PREDICTED: {model_out_text}")

    # BLEU expects list[str] predictions and list[list[str]] references
    metric = torchmetrics.text.BLEUScore()
    bleu   = metric(predicted, [[t] for t in expected])
    wandb.log({"validation/BLEU": bleu, "global_step": global_step})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset download (no HuggingFace account required)
# 用来替代 HuggingFace 的 load_dataset，直接从 OPUS 下载数据
# ─────────────────────────────────────────────────────────────────────────────

def download_opus_books(lang_src, lang_tgt, cache_dir="opus_data"):
    cache_path = Path(cache_dir)
    src_file = cache_path / f"Books.{lang_src}-{lang_tgt}.{lang_src}"
    tgt_file = cache_path / f"Books.{lang_src}-{lang_tgt}.{lang_tgt}"

    if not src_file.exists() or not tgt_file.exists():
        cache_path.mkdir(exist_ok=True)
        url = f"https://object.pouta.csc.fi/OPUS-Books/v1/moses/{lang_src}-{lang_tgt}.txt.zip"
        print(f"Downloading OPUS Books {lang_src}-{lang_tgt} from {url} ...")
        data = requests.get(url).content
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(cache_path)
        print("Download complete.")

    with open(src_file, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f if line.strip()]

    return [
        {"translation": {lang_src: s, lang_tgt: t}}
        for s, t in zip(src_lines, tgt_lines)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_ds(config):
    ds_raw = download_opus_books(config["lang_src"], config["lang_tgt"])
    random.shuffle(ds_raw)

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_size = int(0.9 * len(ds_raw))
    train_raw  = ds_raw[:train_size]
    val_raw    = ds_raw[train_size:]

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt,
                                config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds   = BilingualDataset(val_raw,   tokenizer_src, tokenizer_tgt,
                                config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Diagnostic: show max token lengths in the dataset
    max_len_src = max(len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids) for item in ds_raw)
    max_len_tgt = max(len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids) for item in ds_raw)
    print(f"Max source length: {max_len_src} | Max target length: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader   = DataLoader(val_ds,   batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len, vocab_tgt_len,
        config["seq_len"], config["seq_len"],
        d_model=config["d_model"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    Path(f"{config['datasource'].replace('/', '_')}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    initial_epoch = 0
    global_step   = 0

    # Resume from checkpoint if one exists
    preload        = config["preload"]
    model_filename = (
        latest_weights_file_path(config) if preload == "latest"
        else get_weights_file_path(config, preload) if preload
        else None
    )
    if model_filename and Path(model_filename).exists():
        print(f"Preloading weights from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No checkpoint found — training from scratch")

    # Label smoothing regularizes the model by softening the one-hot target
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"),
        label_smoothing=0.1,
    )

    for epoch in range(initial_epoch, config["num_epochs"]):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iter:
            encoder_input = batch["encoder_input"].to(device)   # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)   # (B, seq_len)
            encoder_mask  = batch["encoder_mask"].to(device)    # (B, 1, 1, seq_len)
            decoder_mask  = batch["decoder_mask"].to(device)    # (B, 1, seq_len, seq_len)
            label         = batch["label"].to(device)           # (B, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output    = model.project(decoder_output)      # (B, seq_len, vocab_size)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )

            batch_iter.set_postfix({"loss": f"{loss.item():.3f}"})
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt,
            config["seq_len"], device,
            lambda msg: batch_iter.write(msg),
            global_step,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            get_weights_file_path(config, f"{epoch:02d}"),
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    config["num_epochs"] = 30
    config["preload"]    = None
    wandb.init(
        project="pytorch-transformer",
        config=config,
    )
    train_model(config)
