from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from src import constants
from src.model import Model
from src.data_loading import DataModule, get_data, get_dicts, get_loss_weight


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    parser.add_argument('--img-dir', default='data/images/sea_floor', help=h)

    parser.add_argument('--accumulate-grad-batches', default=8, type=int, help=h)
    parser.add_argument('--batch-size', default=8, type=int, help=h)
    parser.add_argument('--img-size', default=512, type=int, help=h)
    parser.add_argument('--grayscale', action='store_true', help=h)

    parser.add_argument('--arch', default='mobilenetv3_large_100_miil', help=h)

    parser.add_argument('--folds-to-train', nargs='+', default=[0, 1, 2, 3, 4], type=int, help=h)
    parser.add_argument('--kfolds', default=5, type=int, help=h)
    parser.add_argument('--kfold-seed', default=0, type=int, help=h)
    parser.add_argument('--tta', default=1, type=int, help=h)

    parser.add_argument('--lr', default=1e-4, type=float, help=h)
    parser.add_argument('--lr-factor', default=0.5, type=float, help=h)
    parser.add_argument('--lr-patience', default=1, type=int, help=h)
    parser.add_argument('--stop-patience', default=5, type=int, help=h)
    parser.add_argument('--monitor', default='val_f1', type=str, help=h)
    parser.add_argument('--from-scratch', action='store_true', help=h)

    parser.add_argument('--gpus', default=1, help=h)
    parser.add_argument('--num-workers', default=8, type=int, help=h)

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    mode = 'min' if 'loss' in args.monitor else 'max'
    assert args.kfolds >= len(args.folds_to_train)

    problem = Path(args.img_dir).name

    paths, labels = get_data(args.img_dir, problem)

    if problem == 'sea_floor':
        n_classes = len(set(labels))
        skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.kfold_seed)
    else:
        n_classes = len(labels[0])
        skf = MultilabelStratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.kfold_seed)

    for i, (train_index, val_index) in enumerate(skf.split(paths, labels)):

        print(f'\nTraining {i+1}/{args.kfolds} fold')
        if i not in args.folds_to_train:
            print('Passing')
            continue

        train_dicts = get_dicts(paths, labels, train_index)
        val_dicts = get_dicts(paths, labels, val_index)

        data_module = DataModule(
            train_dicts,
            val_dicts,
            val_dicts,
            args.batch_size,
            args.img_size,
            args.grayscale,
            args.tta,
            args.num_workers
        )

        loss_weight = get_loss_weight([labels[i] for i in train_index], problem)

        model = Model(
            arch=args.arch,
            n_classes=n_classes,
            problem=problem,
            loss_weight=loss_weight,
            lr=args.lr,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            monitor=args.monitor,
            mode=mode,
            pretrained=not args.from_scratch,

            # to be saved in hparams
            img_size=args.img_size,
            grayscale=args.grayscale,
            fold=i+1,
            kfold_seed=args.kfold_seed
        )
        filename = constants.checkpoint_name(problem, args.arch, i+1, args.kfolds, args.monitor)

        trainer = pl.Trainer(
            accumulate_grad_batches=args.accumulate_grad_batches,
            gpus=args.gpus,
            benchmark=True,
            precision=16,
            callbacks=[
                pl.callbacks.ProgressBar(),
                pl.callbacks.EarlyStopping(monitor=args.monitor, patience=args.stop_patience, mode=mode),
                pl.callbacks.ModelCheckpoint(
                    # dirpath='lightning_logs/' + problem + 'aaaa',
                    monitor=args.monitor,
                    mode=mode,
                    filename=filename
                ),
            ],
        )

        print(model.hparams_initial)

        trainer.fit(model, data_module)
