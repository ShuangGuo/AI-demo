import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len      = seq_len
        self.ds           = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang     = src_lang
        self.tgt_lang     = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair     = self.ds[idx]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # How many PAD tokens we need to reach seq_len
        # Encoder: [SOS] tokens [EOS] [PAD...]  → -2 for SOS+EOS
        # Decoder input:  [SOS] tokens [PAD...]  → -1 for SOS
        # Decoder label:  tokens [EOS] [PAD...]  → -1 for EOS
        enc_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_padding = self.seq_len - len(dec_input_tokens) - 1

        if enc_padding < 0 or dec_padding < 0:
            raise ValueError("Sentence is too long — increase seq_len in config")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.expand(enc_padding),
        ])

        # Decoder sees [SOS] + target tokens (teacher forcing); label is shifted left by 1
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.pad_token.expand(dec_padding),
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.expand(dec_padding),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,           # (seq_len,)
            "decoder_input": decoder_input,           # (seq_len,)
            # Encoder padding mask: (1, 1, seq_len) — broadcast over all heads
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # Decoder mask: padding mask AND causal mask combined
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len),
            "label": label,                           # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    """Upper-triangular mask of 1s (including diagonal) flipped so that
    position i can only attend to positions 0..i."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
