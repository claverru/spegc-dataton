import matplotlib.pyplot as plt


def plot_grid(img, labels, probas, heatmaps, out_path='plots/grid.png'):

    plt.clf()

    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

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

    plt.savefig(out_path)
