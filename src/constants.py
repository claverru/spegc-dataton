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


COLORS = {
    'elements': {
        'algas': [0, 255, 0],
        'basura': [255, 0, 0],
        'ripples': [155, 103, 60],
        'fauna': [0, 0, 255],
        'roca': [106, 26, 74]
    }
}

def checkpoint_name(problem, arch, epoch, kfolds, monitor):
    return f'{problem}_{arch}_fold_{epoch}_of_{kfolds}_' + '{epoch}_{' + monitor + ':.3f}'
