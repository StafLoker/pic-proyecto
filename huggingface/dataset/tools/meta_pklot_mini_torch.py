"""PyTorch adapter for meta_pklot_mini. Turns the loader's data into torchvision
datasets; it adds no reading or validation logic.

    from torch.utils.data import DataLoader
    from torchvision.transforms import v2
    from meta_pklot_mini import MetaPKLotMini, split_photos
    from meta_pklot_mini_torch import DetectionDataset, SpotDataset, detection_collate

    train, test = split_photos(MetaPKLotMini('Meta-PKLot-mini'))

    # detection: Faster R-CNN and friends. num_classes = len(DetectionDataset.classes)
    DataLoader(DetectionDataset(train), batch_size=4, collate_fn=detection_collate)

    # classification of cropped spots; ds.targets feeds a WeightedRandomSampler
    ds = SpotDataset(train, transform=v2.Resize((64, 64)))
    DataLoader(ds, batch_size=64, shuffle=True)

`transform` differs on purpose: DetectionDataset passes (image, target) as tensors, so
geometric v2 transforms also move the boxes; SpotDataset passes the PIL crop only.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from meta_pklot_mini import Photo

__all__ = ['DetectionDataset', 'SpotDataset', 'detection_collate']

_SPOT_CLASSES = ('empty', 'occupied')


def detection_collate(batch):
    return tuple(zip(*batch))


def _per_box_fields(sample):
    # Sanitize drops boxes, so every per-box tensor must be filtered with them.
    # Module level (not a lambda) so DataLoader workers can pickle it under spawn.
    target = sample[1]
    return target['labels'], target['area'], target['iscrowd']


class DetectionDataset(Dataset):
    """(image, target) in torchvision format. Image is float32 [0, 1] CxHxW.
    target: boxes XYXY, labels (1 empty, 2 occupied; 0 is the background), image_id,
    area, iscrowd and, with `masks=True`, masks NxHxW (heavy: one mask per spot).
    `transform` is a v2 transform taking (image, target), e.g. RandomHorizontalFlip."""

    # index = label; num_classes = len(classes)
    classes = ('background', *_SPOT_CLASSES)

    def __init__(self, photos: Iterable[Photo], transform: Callable | None = None,
                 *, masks: bool = False):
        self.photos = list(photos)
        self.transform = transform
        self.masks = masks
        steps = [transform] if transform is not None else []
        steps += [v2.ToDtype(torch.float32, scale=True),
                  v2.SanitizeBoundingBoxes(labels_getter=_per_box_fields)]
        self._pipeline = v2.Compose(steps)

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, index: int):
        photo = self.photos[index]
        spots = photo.spots
        size = (photo.height, photo.width)
        image = tv_tensors.Image(decode_image(
            str(photo.path), mode=ImageReadMode.RGB))

        boxes = torch.tensor([s.bbox_xyxy for s in spots],
                             dtype=torch.float32).reshape(-1, 4)
        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=size),
            'labels': torch.tensor([self.classes.index(s.label) for s in spots], dtype=torch.int64),
            'image_id': photo.id,
            'area': torch.tensor([s.area for s in spots], dtype=torch.float32),
            'iscrowd': torch.zeros(len(spots), dtype=torch.int64),
        }
        if self.masks:
            stack = torch.zeros((len(spots), *size), dtype=torch.uint8)
            for k, s in enumerate(spots):
                canvas = Image.new('L', (photo.width, photo.height), 0)
                ImageDraw.Draw(canvas).polygon(s.polygon, fill=1)
                stack[k] = torch.from_numpy(np.array(canvas, dtype=np.uint8))
            target['masks'] = tv_tensors.Mask(stack)
        return self._pipeline(image, target)


class SpotDataset(Dataset):
    """(spot crop, label); label is the index in `classes` and `targets` holds all labels
    without decoding any image, like ImageFolder. `transform` receives the PIL
    crop (Resize, augmentations...). Output is always float32 [0, 1].
    Every access decodes the whole photo: with shuffle, use `num_workers` or
    export the crops to disk."""

    classes = _SPOT_CLASSES

    def __init__(self, photos: Iterable[Photo], transform: Callable | None = None):
        self._items = [(p, s) for p in photos for s in p.spots]
        self.targets = [self.classes.index(s.label) for _, s in self._items]
        self.transform = transform
        steps = [transform] if transform is not None else []
        steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        self._pipeline = v2.Compose(steps)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        photo, spot = self._items[index]
        with Image.open(photo.path) as im:
            # convert only the crop, not the photo
            crop = spot.crop(im).convert('RGB')
        return self._pipeline(crop), self.targets[index]
