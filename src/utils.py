import matplotlib.pyplot as plt

from src.model import Model


def plot_grid(img, labels, probas, heatmaps, sea_floor_label, out_path='plots/grid.png'):

    plt.clf()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    for i, (label, heatmap, p) in enumerate(zip(labels, heatmaps, probas), start=1):

        row = i // 3
        col = i % 3

        ax[row, col].imshow(img, alpha=0.5)
        ax[row, col].axis('off')
        ax[row, col].set_title(label)

        if p > 0.5:
            ax[row, col].imshow(heatmap, cmap='jet', alpha=0.5)

    fig.suptitle(sea_floor_label)

    plt.savefig(out_path)


def load_model(checkpoint_path):
    return Model.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location='cpu').model
