from typing import NamedTuple

import pandas as pd
import torch
from transformers import AutoTokenizer


class PairedSentences(NamedTuple):
    fr: str
    en: str


class ListPairedSentences(NamedTuple):
    fr: list[str]
    en: list[str]

    def __getitem__(self, index: int) -> PairedSentences:
        return PairedSentences(self.fr[index], self.en[index])


class TrainingBatch(NamedTuple):
    input_ids: torch.Tensor
    """In our case french"""

    encoder_mask: torch.Tensor
    """basically padding tokens"""

    output_ids: torch.Tensor
    """In our case english"""

    decoder_mask: torch.Tensor
    """Padding tokens, don't forget to add causal mask during training"""

    def __repr__(self):
        return f"TrainingBatch(x.shape={self.input_ids.shape}, y.shape={self.output_ids.shape})"


class SentenceProcessor:
    def __init__(self, sequence_length: int, tokenizer_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._seq_length = sequence_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def sequence_length(self) -> int:
        return self._seq_length

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def tokenize(
        self,
        text: str,
        padding: str = "max_length",
        truncation: bool = True,
        extra: int = 0,
    ):
        return self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._seq_length + extra,
            padding=padding,
            truncation=truncation,
        )

    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def make_batch(
        self, paired_sentences: ListPairedSentences, dtype=torch.float32
    ) -> TrainingBatch:
        # Tokenize each sentence in the 'fr' and 'en' lists
        fr_sentences = [self.tokenize(sentence) for sentence in paired_sentences.fr]
        en_sentences = [
            self.tokenize(sentence, extra=1) for sentence in paired_sentences.en
        ]

        # Stack tokenized tensors for batching
        X_batch = torch.stack([x["input_ids"].squeeze(0) for x in fr_sentences])
        Y_batch = torch.stack([y["input_ids"].squeeze(0) for y in en_sentences])

        # Create encoder and decoder padding mask: 1 for real tokens, 0 for padding
        encoder_mask = (
            torch.stack([x["attention_mask"].squeeze(0) for x in fr_sentences])
            .unsqueeze(1)
            .unsqueeze(2)
        )
        decoder_mask = (
            torch.stack([y["attention_mask"].squeeze(0) for y in en_sentences])
            .unsqueeze(1)
            .unsqueeze(2)
        )

        return TrainingBatch(
            input_ids=X_batch,
            output_ids=Y_batch,
            encoder_mask=encoder_mask.to(dtype),
            decoder_mask=decoder_mask.to(dtype),
        )


def make_generator_v2(csv_path: str, rows_per_page: int):
    for chunk in pd.read_csv(csv_path, chunksize=rows_per_page):
        fr_sentences = chunk["fr"].fillna("").to_list()
        en_sentences = chunk["en"].fillna("").to_list()
        yield ListPairedSentences(fr_sentences, en_sentences)
