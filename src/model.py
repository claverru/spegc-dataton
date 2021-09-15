import timm
import torch
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self,
                 arch='mobilenetv3_large_100_miil',
                 n_classes=5,
                 lr=1e-4,
                 lr_factor=0.5,
                 lr_patience=1,
                 monitor='val_loss',
                 mode='min',
                 pretrained=True,
                 **kwargs):
        super().__init__()

        self.model = timm.create_model(arch, pretrained=pretrained)
        self.model = self.model.eval()
        self.model.reset_classifier(n_classes)

        self.loss_object = None

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.monitor = monitor
        self.mode = mode

        self.save_hyperparameters()

    def forward(self, x):

        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, target = batch

        pred = self.forward(x)

        loss = self.loss_object(pred, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self.forward(x)
        pass

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
