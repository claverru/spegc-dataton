import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_grid(img,
              labels,
              probas,
              heatmaps,
              sea_floor_label,
              sea_floor_proba,
              img_with_bboxes=None,
              out_path='plots/grid.png'):

    plt.clf()

    fig, ax = plt.subplots(ncols=max(len(labels) + 1, 2), figsize=(20, 10))

    ax[0].imshow(img if img_with_bboxes is None else img_with_bboxes)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    if not labels:
        ax[1].axis('off')
        ax[1].set_title('Nothing predicted')

    for col, (label, heatmap, p) in enumerate(zip(labels, heatmaps, probas), start=1):

        ax[col].imshow(img, alpha=0.5)
        ax[col].axis('off')
        ax[col].set_title(f'{label} ({p:.2f})' )
        ax[col].imshow(heatmap, cmap='jet', alpha=0.5)

    fig.suptitle(f'{sea_floor_label} ({sea_floor_proba:.2f})')

    plt.savefig(out_path)


def get_img_with_bboxes(img, resized_img, labels, heatmaps, clip_agent):

    if 'fauna' not in labels:
        return None

    bboxes = get_norm_bboxes_from_heatmap(heatmaps[labels.index('fauna')])

    if not bboxes:
        return None

    h, w = img.shape[:2]
    fauna_labels = []
    for x_min, y_min, x_max, y_max in bboxes:
        fauna_img = img[int(y_min*h):int(y_max*h), int(x_min*w):int(x_max*w)]
        fauna_pred = clip_agent(fauna_img)
        fauna_label = fauna_pred['label'] + f' ({fauna_pred["label_proba"]:.2f})'
        fauna_labels.append(fauna_label)

    return draw_bboxes(resized_img, bboxes, fauna_labels)


def get_norm_bboxes_from_heatmap(heatmap):
    h, w = heatmap.shape[-2:]
    pos = heatmap[heatmap > 0]
    binary_hm = np.where(heatmap > pos.mean(), 1, 0).astype(np.uint8)
    n_classes, cc = cv2.connectedComponents(binary_hm)
    bboxes = []
    for i in range(1, n_classes):
        ys, xs = np.where(cc == i)
        xs = xs / w
        ys = ys / h
        x_min, y_min, x_max, y_max = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        if x_max - x_min > 0.1 and y_max - y_min > 0.1:
            bboxes.append((x_min, max(y_min, 0.05), x_max, y_max))
    return bboxes


def draw_bboxes(img, norm_bboxes, labels, color=(255, 0, 0)):
    h, w = img.shape[:2]
    img = img.copy()
    for (x_min, y_min, x_max, y_max), label in zip(norm_bboxes, labels):

        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 4)
        cv2.putText(img, label[:30], (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def bring_back(tensor_dict):
    if isinstance(tensor_dict, dict):
        for k in tensor_dict:
            tensor_dict[k] = bring_back(tensor_dict[k])
        return tensor_dict
    else:
        return tensor_dict.squeeze(0).cpu().numpy()
