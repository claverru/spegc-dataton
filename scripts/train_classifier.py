from argparse import ArgumentParser

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from src.model import Model


def get_parser():
    parser = ArgumentParser()
    # TODO: Add help

    parser.add_argument('--img-dir', default='data/classification/images')
    parser.add_argument('--labels-path', default='data/classification/labels.json')

    parser.add_argument('--accumulate-grad-batches', default=8, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--img-size', default=512, type=int)
    parser.add_argument('--arch', default='mobilenetv3_large_100_miil')

    parser.add_argument('--folds-to-train', nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--kfolds', default=5, type=int)
    parser.add_argument('--kfold-seed', default=0, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr-factor', default=0.5, type=float)
    parser.add_argument('--lr-patience', default=1, type=int)
    parser.add_argument('--monitor', default='val_loss', type=str)
    parser.add_argument('--from-scratch', action='store_true')

    parser.add_argument('--gpus', default=1)

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    mode = 'min' if 'loss' in args.monitor else 'max'
    assert args.kfolds >= len(args.folds_to_train)

    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    sfk = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.kfold_seed)
    for i, (train_index, test_index) in enumerate(sfk.split(X, y)):

        print(f'\nTraining {i+1}/{args.kfolds} fold')
        if i not in args.folds_to_train:
            print('Passing')
            continue

        model = Model(
            arch='mobilenetv3_large_100_miil',
            n_classes=10, # TODO: define
            lr=args.lr,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            monitor=args.monitor,
            mode=mode,
            pretrained=not args.from_scratch,

            # to be saved in hparams
            img_size=args.img_size,
            fold=i+1
        )

        trainer = pl.Trainer(
            accumulate_grad_batches=args.accumulate_grad_batches,
            gpus=args.gpus,
            benchmark=True,
            precision=16,
            callbacks=[
                pl.callbacks.ProgressBar(),
                pl.callbacks.EarlyStopping(monitor=args.monitor, patience=args.lr_patience, mode=mode),
                pl.callbacks.ModelCheckpoint(
                    monitor=args.monitor,
                    mode=mode,
                    filename=f'{args.arch}_fold_{i+1}_of_{args.kfolds}_' + '{epoch}_{args.monitor:.3f}'),
            ],
        )

        print(model.hparams)