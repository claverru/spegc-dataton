import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_heatmap(img, pred, color, alpha=0.3, grayscale=True):
    if grayscale:
        multiplier = np.array([0.1140, 0.5870, 0.2989])[None, None, :]
        img = (img * multiplier).sum(-1, keepdims=True)
    red_pred = pred[..., None] * np.array(color)
    # blue_pred = (pred >= 0.5)[..., None] * np.array([0, 0, 255])

    # blue_img = np.where(
    #     blue_pred.any(-1, keepdims=True),
    #     (1 - alpha) * img + alpha * blue_pred,
    #     img
    # ).astype(np.float32)
    red_img = np.where(
        red_pred.any(-1, keepdims=True),
        (1 - alpha) * img + alpha * red_pred,
        img
    ).astype(np.uint8)

    return red_img
