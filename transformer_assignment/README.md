# Transformer from Scratch — EN→IT Translation

A minimal Transformer (encoder + decoder) implemented in PyTorch for English-to-Italian
translation using the OPUS Books dataset.

## Project Structure

```
transformer_assignment/
├── config.py        # Hyperparameters and checkpoint paths
├── model.py         # All Transformer modules from scratch
├── dataset.py       # BilingualDataset + causal_mask
├── train_wb.py      # Training loop (W&B logging + BLEU validation)
├── translate.py     # CLI inference on a trained model
└── requirements.txt
```

## Setup

Python 3.14 is required. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

No HuggingFace account is needed. The dataset is downloaded automatically from the
[OPUS website](https://opus.nlpl.eu) on first run and cached locally in `opus_data/`.

## Training

```bash
python train_wb.py
```

Checkpoints are saved to `Helsinki-NLP_opus_books_weights/tmodel_XX.pt` after each epoch.
Training loss and validation BLEU are logged to Weights & Biases.

**On Mac (Apple Silicon):** training automatically uses the MPS (GPU) backend, which is
roughly 20x faster than CPU. To prevent the machine from sleeping mid-run:

```bash
caffeinate -i python train_wb.py
```

Training can be interrupted at any time with Ctrl+C and will resume from the latest
checkpoint on the next run.

## Translation (after training)

```bash
# Translate a custom sentence
python translate.py "The book was very interesting."

# Use dataset index to translate a known example
python translate.py 42
```

## Design Choices

| Choice | Reason |
|--------|--------|
| Pre-Norm (LayerNorm before sublayer) | More stable gradients vs Post-Norm |
| Xavier uniform init | Prevents vanishing/exploding gradients at start |
| Label smoothing = 0.1 | Regularizes over-confident softmax predictions |
| WordLevel tokenizer | Simple, interpretable; sufficient for this dataset size |
| ignore_index=[PAD] in loss | Padding tokens do not contribute to gradient |
| Greedy decode at validation | Fast; BLEU reported as a proxy for translation quality |
| OPUS direct download | Avoids HuggingFace authentication requirement |

## Module Architecture

```
InputEmbeddings + PositionalEncoding
        |
   +----v----+  x N
   |Encoder  |  LayerNorm -> MultiHeadAttention -> Residual
   |Block    |  LayerNorm -> FeedForward         -> Residual
   +----+----+
        | encoder_output
   +----v----+  x N
   |Decoder  |  LayerNorm -> Masked Self-Attention  -> Residual
   |Block    |  LayerNorm -> Cross-Attention         -> Residual
   +----+----+  LayerNorm -> FeedForward             -> Residual
        |
   ProjectionLayer -> vocab logits -> CrossEntropyLoss
```

## Observed Results

Trained on OPUS Books (en-it), default config (d_model=512, 6 layers, 8 heads, 20 epochs):

| Epoch | Training Loss |
|-------|--------------|
| 0     | ~10.0        |
| 4     | ~4.5         |
| 9     | ~2.5 (est.)  |
| 19    | ~1.2 (est.)  |

- Device: Apple M-series (MPS), ~16 min per epoch
- Validation BLEU improves steadily across epochs
- Early epochs produce mostly random Italian words; later epochs show coherent phrases

## Configuration

Key hyperparameters in `config.py`:

| Parameter   | Value | Description               |
|-------------|-------|---------------------------|
| d_model     | 512   | Model embedding dimension |
| num_epochs  | 20    | Total training epochs     |
| batch_size  | 8     | Batch size                |
| seq_len     | 350   | Maximum sequence length   |
| lr          | 1e-4  | Adam learning rate        |
| lang_src    | en    | Source language           |
| lang_tgt    | it    | Target language           |
