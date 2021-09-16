import timm
import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self,
                 arch='mobilenetv3_large_100_miil',
                 classes={'sea_floor': 3, 'elements': 4},
                 loss_weights={'sea_floor': 1, 'elements': 1},
                 lr=1e-4,
                 lr_factor=0.5,
                 lr_patience=1,
                 monitor='val_loss',
                 mode='min',
                 pretrained=True,
                 **kwargs):
        super().__init__()

        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.backbone = self.backbone.eval()

        self.classifiers = {
            'sea_floor': torch.nn.Linear(in_features=self.backbone.num_features, out_features=classes['sea_floor']),
            'elements': torch.nn.Linear(in_features=self.backbone.num_features, out_features=classes['elements']),
        }

        self.loss_objects = {
            'sea_floor': torch.nn.CrossEntropyLoss() if classes['sea_floor'] > 1 else torch.nn.MSELoss(),
            'elements': torch.nn.BCEWithLogitsLoss() # TODO: investigate weighed multilabel loss
        }

        self.loss_weights = loss_weights
        # TODO: add metrics

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.monitor = monitor
        self.mode = mode

        self.save_hyperparameters()

    def forward(self, x):
        features = self.backbone(x)
        return {
            'sea_floor': self.classifiers['sea_floor'](features),
            'elements': self.classifiers['elements'](features)
        }

    def training_step(self, batch, batch_idx):
        x, target = batch

        preds = self.forward(x)

        loss = sum(self.loss_objects[c](preds[c], target[c]) * w for c, w in self.loss_weights.items())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        x, target = batch

        preds = self.forward(x)

        loss = sum(self.loss_objects[c](preds[c], target[c]) * w for c, w in self.loss_weights.items())
        # TODO: Add metrics

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
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
