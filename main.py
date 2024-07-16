from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
import torch
from models import ABAW7Models
from dataloader import ABAW7DataModule
import os


class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='epoch'):
        super().__init__(write_interval)
        print('Writing prediction to file')
        self.output_dir = output_dir

    def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


def cli_main():
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(ABAW7Models, ABAW7DataModule)


if __name__ == '__main__':
    cli_main()
