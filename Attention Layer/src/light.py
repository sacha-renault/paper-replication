import typing as tp

import lightning as L
import torch
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.processor import SentenceProcessor
from src.transformer import Transformer


class LitTransformer(L.LightningModule):
    def __init__(
        self,
        transformer: Transformer,
        processor: SentenceProcessor,
        *args,
        optimizer: tp.Type[Optimizer] | None = None,
        optimizer_kwargs: tp.Dict[str, tp.Any] | None = None,
        scheduler: tp.Callable[[tp.Any], LambdaLR] | None = None,
        scheduler_kwargs: tp.Dict[str, tp.Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        pad_token = processor.tokenizer.pad_token_id  # type: ignore
        self._model = transformer
        self._loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
        self._proc = processor
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._optimizer_kwargs = optimizer_kwargs
        self._scheduler_kwargs = scheduler_kwargs

    def training_step(self, batch) -> torch.Tensor | None:
        # make a batch
        try:
            training_batch = self._proc.make_batch(batch)
        except Exception:
            # Log exception ?
            return None

        # Encoder input (source sentence)
        input_ids = training_batch.input_ids.to(self.device)
        encoder_mask = training_batch.encoder_mask.to(self.device)

        # adjust decoder input
        target_ids = training_batch.output_ids.to(self.device)
        decoder_input_ids = target_ids[:, :-1]
        target_ids_flat = target_ids[:, 1:].contiguous().view(-1)

        # adjust decoder mask too
        decoder_mask = training_batch.decoder_mask[:, :, :, :-1].to(self.device)

        # We need to create a causal mask too
        seq_len = decoder_input_ids.shape[1]
        causal_mask = (
            torch.tril(torch.ones((seq_len, seq_len)))
            .to(device=self.device)
            .to(self.dtype)
        )

        # final decoder mask as prod of padding * causal
        final_decoder_mask = decoder_mask * causal_mask.unsqueeze(0)

        # Forward pass
        output_probs = self._model(
            input_ids,
            decoder_input_ids,
            encoder_mask=encoder_mask,
            decoder_mask=final_decoder_mask,
        )

        # flatten target and outputprobs to compute cce loss
        output_probs_flat = output_probs.view(-1, output_probs.size(-1))

        # Calculate the loss
        loss = self._loss_fn(output_probs_flat, target_ids_flat)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("loss", loss, True)
        self.log("lr", lr, True)
        return loss

    def configure_optimizers(self):
        if self._optimizer is not None:
            optimizer = self._optimizer(
                self._model.parameters(), **(self._optimizer_kwargs or {})
            )
        else:
            optimizer = optim.Adam(self._model.parameters(), lr=3e-4)

        if self._lr_scheduler is None:
            return optimizer
        else:
            scheduler = self._lr_scheduler(optimizer, **(self._scheduler_kwargs or {}))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    @torch.no_grad()
    def translate(self, text: str, skip_special_tokens: bool = True) -> str:
        # first model as eval (we don't train here)
        sequence_length = self._proc.sequence_length

        # Tokenize the input sequence in french
        tokens = self._proc.tokenize(text)

        # create the encoder input and mask
        encoder_ids = tokens["input_ids"].int().to(self.device)
        encoder_mask = tokens["attention_mask"].unsqueeze(0).to(self.device)

        # get dtype
        dtype: torch.dtype = self.dtype  # type: ignore

        # Create padding mask for current sequence length
        padding_mask = torch.zeros((1, 1, 1, sequence_length), dtype=dtype).to(
            self.device
        )
        causal_mask = (
            torch.tril(torch.ones((sequence_length, sequence_length), dtype=dtype))
            .unsqueeze(0)
            .to(self.device)
        )

        # Initialize decoder input with just the start token
        generated_ids = torch.zeros((1, sequence_length), dtype=torch.int).to(
            self.device
        )
        generated_ids[0, 0] = self._proc.tokenizer.cls_token_id  # type: ignore

        # loop to generate output ids
        for idx in range(1, sequence_length):
            padding_mask[0, 0, :idx, :idx] = (
                1.0  # All positions up to current length are unmasked
            )
            final_mask = padding_mask * causal_mask
            # return
            output_probs = self._model(
                encoder_ids,
                generated_ids,
                encoder_mask=encoder_mask,
                decoder_mask=final_mask,
            )

            # Select next token from the LAST position (current_len - 1)
            next_token_id = output_probs[0, idx - 1, :].argmax(
                dim=-1
            )  # Last position, all vocab

            # Append the new token
            generated_ids[0, idx] = next_token_id

            # early stop when encounter sep_token_id
            if next_token_id.item() == self._proc.tokenizer.sep_token_id:  # type: ignore
                break

        return self._proc.decode(
            generated_ids[0], skip_special_tokens=skip_special_tokens
        )
