from lightning import LightningModule
from torchmetrics import Accuracy
from torchvision.models import regnet_y_400mf, mobilenet_v3_large
from torch import nn
from loss import CCCLoss, CEFocalLoss, SigmoidFocalLoss, MSELoss

from functools import partial
from torchmetrics import F1Score
from metrics import ConCorrCoef
import torch
from torch.nn import functional as F


class ABAWDecoder(nn.Module):
    def __init__(self, n_features, n_outputs, n_layers):
        super(ABAWDecoder, self).__init__()
        layers = []
        n_outs = n_features
        for idx in range(n_layers - 1):
            n_outs = max(2 ** (8 - idx), 64)
            layers.append(nn.Linear(in_features=n_features, out_features=n_outs))
            layers.append(nn.ReLU())
            n_features = n_outs

        layers.append(nn.Linear(in_features=n_outs, out_features=n_outputs))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class ABAW7Models(LightningModule):
    def __init__(self, label_smoothing=0., alpha=0.75, gamma=2):
        super().__init__()

        self.save_hyperparameters()

        self.encoder_reg = regnet_y_400mf(weights='DEFAULT')

        n_features_reg = self.encoder_reg.fc.in_features

        self.encoder_reg.fc = nn.Identity()

        n_features = n_features_reg
        self.decoder = nn.ModuleDict({
            'va': ABAWDecoder(n_features, n_outputs=2, n_layers=4),
            'expr': ABAWDecoder(n_features, n_outputs=8, n_layers=3),
            'aus': ABAWDecoder(n_features, n_outputs=12, n_layers=4)
        })

        self.loss_module = {'va': partial(CCCLoss, num_classes=2),
                            'expr': partial(CEFocalLoss, label_smoothing=label_smoothing,
                                            alpha=alpha, gamma=gamma),
                            'aus': partial(SigmoidFocalLoss, num_classes=12,
                                           alpha=alpha, gamma=gamma)}

        self.train_metrics_module = nn.ModuleDict({'va': ConCorrCoef(num_classes=2),
                                                   'expr': F1Score(task='multiclass', num_classes=8, average='macro'),
                                                   'aus': F1Score(task='multilabel', num_labels=12, average='macro')})
        self.eval_metrics_module = nn.ModuleDict({'va': ConCorrCoef(num_classes=2),
                                                  'expr': F1Score(task='multiclass', num_classes=8, average='macro'),
                                                  'aus': F1Score(task='multilabel', num_labels=12, average='macro')})

        self.loss_weights = {'va': 1.0, 'expr': 0.0, 'aus': 0.0}

    def forward(self, batchs):
        x = batchs['image']
        z = self.encoder_reg(x)

        pred = {}
        for k in ['va', 'expr', 'aus']:
            pred[k] = self.decoder[k](z)

        return pred

    def training_step(self, batch, batch_idx):
        z, loss_dict = self._shared_eval(batch, batch_idx, 'train')

        return loss_dict['train/loss']

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch)

        y_hat['expr'] = torch.argmax(F.softmax(y_hat['expr'], dim=1), dim=1, keepdim=True)
        y_hat['aus'] = F.sigmoid(y_hat['aus'])

        return y_hat, batch

    def _shared_eval(self, batch, batch_idx, prefix):
        z = self.forward(batch)

        loss_dict = {}
        for k in ['va', 'expr', 'aus']:
            get_indexes = self.ignore_unlabeled_data(k, batch[k])
            loss_dict[f'{prefix}/loss_{k}'] = self.loss_module[k](z[k][get_indexes, :],
                                                                  batch[k][get_indexes, :])  # * self.loss_weights[k]
            if k == 'expr':
                batch[k] = torch.argmax(batch[k], dim=1, keepdim=False)
            elif k == 'aus':
                z[k] = F.sigmoid(z[k])

            if prefix == 'train':
                self.train_metrics_module[k].update(z[k][get_indexes, :], batch[k][get_indexes])
            else:
                self.eval_metrics_module[k].update(z[k][get_indexes, :], batch[k][get_indexes])

        total_loss = sum(loss_dict.values())
        loss_dict[f'{prefix}/loss'] = total_loss

        # self.log_dict(loss_dict, prog_bar=True, logger=True)
        return z, loss_dict

    @staticmethod
    def ignore_unlabeled_data(task, arr, ):
        if task == 'va':
            return torch.sum(arr, dim=1, keepdim=False) > -2
        elif task in ['expr', 'aus']:
            return torch.sum(arr, dim=1, keepdim=False) > 0
        else:
            raise ValueError(f'Unknown task: {task}')

    def on_train_epoch_end(self):
        metrics_dict = {}
        for k in ['va', 'expr', 'aus']:
            metrics_dict[f'train/{k}_score'] = self.train_metrics_module[k].compute()
            self.train_metrics_module[k].reset()

        total_score = sum(metrics_dict.values())
        metrics_dict['train/score'] = total_score

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics_dict = {}
        for k in ['va', 'expr', 'aus']:
            metrics_dict[f'val/{k}_score'] = self.eval_metrics_module[k].compute()

            self.eval_metrics_module[k].reset()

        total_score = sum(metrics_dict.values())
        metrics_dict['val/score'] = total_score

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
