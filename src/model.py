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
