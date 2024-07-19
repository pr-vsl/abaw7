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
    def __init__(self, n_features, n_outputs, n_chunks):
        super(ABAWDecoder, self).__init__()
        assert n_features % n_chunks == 0
        self.n_features = n_features
        self.n_chunks = n_chunks

        if self.n_chunks > 1:
            self.latent_decode = nn.Sequential(nn.Linear(self.n_chunks, self.n_chunks // 4), nn.ReLU(),
                                               nn.Linear(self.n_chunks // 4, self.n_chunks), nn.Sigmoid()
                                               )

        self.fc = nn.Linear(self.n_features, n_outputs)

    def forward(self, x):
        if self.n_chunks > 1:
            x = torch.reshape(x, (-1, self.n_features // self.n_chunks, self.n_chunks))
            x_wgt = self.latent_decode(x)
            x = x * x_wgt
            x_weighted = torch.reshape(x, (-1, self.n_features))
        else:
            x_weighted = x
        x = self.fc(x_weighted)
        return x, x_weighted


class ABAW7Models(LightningModule):
    def __init__(self, label_smoothing=0., alpha=0.75, gamma=2, dropout_rate=0.3):
        super().__init__()

        self.save_hyperparameters()

        self.encoder_reg = regnet_y_400mf(weights='DEFAULT')

        n_features_reg = self.encoder_reg.fc.in_features

        self.encoder_reg.fc = nn.Identity()

        n_features = n_features_reg
        self.decoder = nn.ModuleDict({
            'va': ABAWDecoder(n_features, n_outputs=2, n_chunks=1),
            'expr': ABAWDecoder(n_features, n_outputs=8, n_chunks=1),
            'aus': ABAWDecoder(n_features, n_outputs=12, n_chunks=8),
            'va_expr': ABAWDecoder(n_features, n_outputs=2 + 8, n_chunks=1),
            'aus_va': ABAWDecoder(n_features, n_outputs=12 + 2, n_chunks=8),
            'aus_expr': ABAWDecoder(n_features, n_outputs=12 + 8, n_chunks=8),
            'aus_va_expr': ABAWDecoder(n_features, n_outputs=12 + 2 + 8, n_chunks=8),
        })

        self.au_va_latent_dropout = nn.Dropout(dropout_rate)
        self.au_expr_latent_dropout = nn.Dropout(dropout_rate)
        self.au_va_expr_latent_dropout = nn.Dropout(dropout_rate)

        self.fc = nn.ModuleDict({
            'va': nn.Linear(2 * 4, 2),
            'expr': nn.Linear(8 * 4, 8),
            'aus': nn.Linear(12 * 4, 12),
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

        self.loss_weights = {'va': 1.0, 'expr': 1.0, 'aus': 1.0}

    def forward(self, batchs):
        x = batchs['image']
        z = self.encoder_reg(x)

        pred = {}

        aus, latent = self.decoder['aus'](z)
        z_expr = z + self.au_expr_latent_dropout(latent)
        z_au = z + self.au_va_latent_dropout(latent)
        z_va_expr = z + self.au_va_expr_latent_dropout(latent)

        expr, _ = self.decoder['expr'](z_expr)
        va, _ = self.decoder['va'](z_au)

        aus_va, latent_ava = self.decoder['aus_va'](z)
        aus_expr, latent_axpr = self.decoder['aus_expr'](z)
        aus_va_expr, latent_avxpr = self.decoder['aus_va_expr'](z)
        va_expr, _ = self.decoder['va_expr'](z_va_expr)

        pred['va'] = self.fc['va'](torch.concat([va, aus_va[:, :2], aus_va_expr[:, 12:14], va_expr[:, :2]], dim=1))
        pred['expr'] = self.fc['expr'](
            torch.concat([expr, aus_expr[:, 12:], aus_va_expr[:, 14:], va_expr[:, 2:]], dim=1))
        pred['aus'] = self.fc['aus'](torch.concat([aus, aus_va[:, 2:], aus_va_expr[:, :12], aus_expr[:, :12]], dim=1))

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
        y_hat['aus'] = 1 * (F.sigmoid(y_hat['aus']) >= 0.45)

        return y_hat, batch

    def _shared_eval(self, batch, batch_idx, prefix):
        z = self.forward(batch)

        loss_dict = {}
        for k in ['va', 'expr', 'aus']:
            get_indexes = self.ignore_unlabeled_data(k, batch[k])
            loss_dict[f'{prefix}/loss_{k}'] = self.loss_module[k](z[k][get_indexes, :],
                                                                  batch[k][get_indexes, :]) #* self.loss_weights[k]
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

        # self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics_dict = {}
        for k in ['va', 'expr', 'aus']:
            metrics_dict[f'val/{k}_score'] = self.eval_metrics_module[k].compute()
            self.loss_weights[k] = 0.56 / metrics_dict[f'val/{k}_score']
            self.eval_metrics_module[k].reset()

        total_score = sum(metrics_dict.values())
        metrics_dict['val/score'] = total_score

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        metrics_dict = {}
        for k in ['va', 'expr', 'aus']:
            metrics_dict[f'test/{k}_score'] = self.eval_metrics_module[k].compute()
            self.loss_weights[k] = 0.56 / metrics_dict[f'test/{k}_score']
            self.eval_metrics_module[k].reset()

        total_score = sum(metrics_dict.values())
        metrics_dict['test/score'] = total_score

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
