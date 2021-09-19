from typing import Dict, List

import timm
import torch
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self,
                 arch='mobilenetv3_large_100_miil',
                 n_classes=4,
                 problem='sea_floor',
                 loss_weight=None,
                 lr=1e-4,
                 lr_factor=0.5,
                 lr_patience=1,
                 monitor='val_f1',
                 mode='min',
                 pretrained=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.problem = problem

        self.model = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)
        self.model = self.model.eval()

        if loss_weight is not None:
            loss_weight = torch.tensor(loss_weight)

        if problem == 'sea_floor':
            self.loss_object = torch.nn.CrossEntropyLoss(weight=loss_weight)
        elif problem == 'elements':
            self.loss_object = torch.nn.BCEWithLogitsLoss(weight=loss_weight)
        else:
            raise ValueError('sea_floor or elements')

        self.f1_object = torchmetrics.F1(n_classes, average='weighted')

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.monitor = monitor
        self.mode = mode

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        preds = self.forward(x)

        loss = self.loss_object(preds, target if self.problem == 'sea_floor' else target.float())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        preds = self.forward(x)

        loss = self.loss_object(preds, target if self.problem == 'sea_floor' else target.float())
        f1 = self.f1_object(preds, target)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_factor,
                patience=self.lr_patience,
                mode=self.mode,
                verbose=True),
            'monitor': self.monitor
        }

    @classmethod
    def load_model(cls, checkpoint_path):
        return cls.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location='cpu').model


class ElementsHeatmapEnsemble(torch.nn.Module):

    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outs = []
        heatmaps = []
        for m in self.models:
            features = m.forward_features(x)
            pooled = m.global_pool(features)
            out = m.fc(pooled)

            inter_features = torch.nn.functional.interpolate(
                features,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            """ Dimensions:
                - c: classes
                - f: features
                - b: batch
                - w: width
                - h: height
            """
            heatmap = torch.einsum('cf, bfwh -> bcwh', m.fc.weight, inter_features)

            outs.append(out)
            heatmaps.append(heatmap)

        return {
            'probas': sum(torch.sigmoid(out) for out in outs)/len(outs),
            'heatmaps': sum(heatmap)/len(heatmaps)
        }


class SeafloorEnsemble(torch.nn.Module):

    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outs = [m(x) for m in self.models]
        return {'probas': sum(torch.softmax(out, -1) for out in outs)/len(outs)}


class MegaEnsemble(torch.nn.Module):

    def __init__(self, models_dict: Dict[str, List[torch.nn.Module]]):
        super().__init__()
        self.models = torch.nn.ModuleDict({
            'elements': ElementsHeatmapEnsemble(models_dict['elements']),
            'sea_floor': SeafloorEnsemble(models_dict['sea_floor'])
        })

    def forward(self, x):
        return {problem: m(x) for problem, m in self.models.items()}
