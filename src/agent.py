import enum
from typing import Dict, List

import clip
import torch
import numpy as np

from src import utils
from src.constants import NAME2ID
from src.model import Model, MegaEnsemble
from src.data_loading import get_basic_transforms, get_clip_transforms


class MegaEnsembleAgent(object):

    def __init__(self,
                 checkpoint_paths: Dict[str, List[str]],
                 img_size: int = 512,
                 elements_threshold: float = 0.5,
                 device: str = 'cpu'):
        self.device = device
        self.img_size = img_size
        self.elements_thresholds = elements_threshold
        self.model = MegaEnsemble({
            'elements': [Model.load_model(cp) for cp in checkpoint_paths['elements']],
            'sea_floor': [Model.load_model(cp) for cp in checkpoint_paths['sea_floor']]
        }).eval().to(self.device)
        self.transforms = get_basic_transforms(self.img_size)
        self.elements_labels = list(NAME2ID['elements'])
        self.sea_floor_labels = list(NAME2ID['sea_floor'])

    def __call__(self, image: np.ndarray):

        X = self.transforms(image=image)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(X)
            preds = utils.bring_back(preds)

        filter_ids = [i for i, p in enumerate(preds['elements']['probas']) if p > self.elements_thresholds]
        for k in list(preds['elements']):
            preds['elements'][f'filtered_{k}'] = preds['elements'][k][filter_ids]
        preds['elements']['filtered_labels'] = [self.elements_labels[i] for i in filter_ids]

        preds['sea_floor']['label'] = self.sea_floor_labels[preds['sea_floor']['probas'].argmax()]
        preds['sea_floor']['proba'] = preds['sea_floor']['probas'].max().round(2)

        return preds



class ClipAgent(object):

    def __init__(self, classes: List[str], *args, device='cpu', **kwargs):
        self.device = device
        self.clip_model, _ = clip.load(*args, device=self.device, **kwargs)
        self.transforms = get_clip_transforms(self.clip_model.visual.input_resolution)
        self.classes = classes
        self.text = clip.tokenize(classes).to(self.device)

    def __call__(self, image: np.ndarray):
        X = self.transforms(image=image)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.clip_model(X, self.text)
            probas = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        return {'probas': probas, 'label': self.classes[probas.argmax()], 'label_proba': probas.max()}
