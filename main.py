from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.cli import LightningCLI
import torch
from models import ABAW7Models
from dataloader import ABAW7DataModule
import os
import numpy as np


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

        pred_dict = {'va': [], 'expr': [], 'aus': []}
        fname = []
        for idx in range(len(predictions)):
            fname.append(predictions[idx][1]['name'])
            for k in ['va', 'expr', 'aus']:
                pred_dict[k].append(predictions[idx][0][k])


        for k in ['va', 'expr', 'aus']:
            pred_dict[k] = torch.concat(pred_dict[k]).to(torch.float).numpy()

        write_arr = np.concatenate((pred_dict['va'], pred_dict['expr'], pred_dict['aus']), axis=1)
        fname = np.concatenate(fname)
        with open(os.path.join(self.output_dir, 'predictions.txt'), "w") as f:
            f.write('image,valence,arousal,expression,aus\n')
            for ix in range(len(fname)):
                write_str = fname[ix] + ',' + ','.join(str(x) for x in write_arr[ix]) + '\n'
                f.write(write_str)


def cli_main():
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(ABAW7Models, ABAW7DataModule)


if __name__ == '__main__':
    cli_main()
