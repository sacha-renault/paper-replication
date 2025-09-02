import lightning as L
import torch

from src.light import LitTransformer
from src.processor import SentenceProcessor, make_generator_v2
from src.transformer import Transformer


class CSVDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_path: str, batch_size: int):
        self.csv_path = csv_path
        self.batch_size = batch_size

    def __iter__(self):
        return make_generator_v2(self.csv_path, self.batch_size)


if __name__ == "__main__":
    # params
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    num_heads = 8
    d_model = 512
    max_len = 200
    batch_size = 48
    num_data = int(48)
    num_batches = num_data // batch_size
    num_epochs = 100

    # dataloader
    def dataloader():
        for _ in range(50):
            yield 0

    # transformer
    processor = SentenceProcessor(max_len, "bert-base-uncased")
    transformer = (
        Transformer(
            num_heads,
            d_model,
            max_sequence_len=processor.sequence_length,
            vocab_size=processor.vocab_size,
        )
        .to(device)
        .to(dtype)
    )
    model = LitTransformer(transformer, processor)

    trainer = L.Trainer(
        limit_train_batches=num_batches,
        max_epochs=num_epochs,
        accumulate_grad_batches=4,
    )
    trainer.fit(
        model=model,
        train_dataloaders=CSVDataset("archive/en-fr.csv", batch_size),
    )
