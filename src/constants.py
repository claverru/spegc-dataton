NAME2ID = {
    'sea_floor': {
        'arenofangoso': 0,
        'arenoso': 1,
        'fangoso': 2,
        'arrecife': 3,
    },
    'elements': {
        'algas': 0,
        'basura': 1,
        'ripples': 2,
        'fauna': 3,
        'roca': 4,
    }
}


def checkpoint_name(problem, arch, epoch, kfolds, monitor):
    return f'{problem}_{arch}_fold_{epoch}_of_{kfolds}_' + '{epoch}_{' + monitor + ':.3f}'
