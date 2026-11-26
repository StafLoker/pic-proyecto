"""Validated reader for Meta-PKLot-mini. Standard library only; Pillow is imported
when needed (checking image sizes, opening images). Data comes out as plain, immutable
types so any framework can use it; the PyTorch adapter lives in meta_pklot_mini_torch.

    from meta_pklot_mini import MetaPKLotMini, split_photos

    ds = MetaPKLotMini('Meta-PKLot-mini')              # validates everything on load
    ds.photos(dataset='PKLot', climate='Sunny')         # filters are keyword-only
    ds.photo(1)                                         # by COCO image id
    train, test = split_photos(ds, test_size=0.2)       # leak-free by day

    for photo in train:
        for spot in photo.spots:
            spot.label, spot.bbox_xyxy, spot.polygon
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Iterable

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    'MetaPKLotMini', 'Photo', 'Spot', 'split_photos',
    'DatasetError', 'AnnotationsNotFoundError', 'SchemaError', 'IntegrityError',
    'GeometryError', 'ImageFileError', 'UnknownNameError',
]

_ERRORS_SHOWN = 10
# Coordinates are rounded to 2 decimals, so exact comparisons would reject good data.
_COORD_TOL = 0.02
_LOOKUPS = ('categories', 'climates', 'times_of_day',
            'conditions', 'spot_types', 'datasets')
_REQUIRED_KEYS = {
    'images': {'id', 'file_name', 'width', 'height', 'subset_id', 'date', 'hour',
               'time_of_day_id', 'climate_id', 'condition_ids', 'original'},
    'annotations': {'id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox',
                    'truncated', 'spot_type_id'},
    'subsets': {'id', 'dataset_id', 'parking', 'camera'},
    **{name: {'id', 'name'} for name in _LOOKUPS},
}


class DatasetError(Exception):
    """Base class. `problems` lists every failure of the stage, not just the first."""

    def __init__(self, problems: list[str]):
        shown = '\n  '.join(problems[:_ERRORS_SHOWN])
        more = len(problems) - _ERRORS_SHOWN
        tail = f'\n  ... and {more} more' if more > 0 else ''
        self.problems = problems
        self.message = f'{type(self).__name__}: {len(problems)} problem(s)\n  {shown}{tail}'
        super().__init__(self.message)


class SchemaError(DatasetError):
    """Unreadable JSON or missing keys."""


class IntegrityError(DatasetError):
    """Duplicate ids, ids pointing to nothing, out-of-range values."""


class GeometryError(DatasetError):
    """Degenerate or non-finite polygons and bboxes, or ones outside the image."""


class ImageFileError(DatasetError):
    """Image missing, unreadable, or with a size different from the annotated one."""


class AnnotationsNotFoundError(DatasetError):
    """`annotations.json` is not in the dataset root."""


class UnknownNameError(ValueError):
    """A filter used a name the dataset does not have. Not a DatasetError: the data
    is fine, the call is wrong."""


def _fail_on(problems: list[str], error: type[DatasetError]) -> None:
    if problems:
        raise error(problems)


def _check_schema(coco: dict) -> None:
    problems = [f'missing key {k!r}' for k in sorted(
        set(_REQUIRED_KEYS) - coco.keys())]

    if not problems:
        for table, keys in _REQUIRED_KEYS.items():
            problems += [
                f'{table} id={row.get("id")}: missing {sorted(keys - row.keys())}'
                for row in coco[table]
                if keys - row.keys()
            ]
    if not problems:
        problems += [
            f'images id={i["id"]}: original needs file_name and crop'
            for i in coco['images']
            if not isinstance(i['original'], dict)
            or {'file_name', 'crop'} - i['original'].keys()
        ]
    _fail_on(problems, SchemaError)


def _check_integrity(coco: dict) -> None:
    problems: list[str] = []
    ids = {t: Counter(r['id'] for r in coco[t]) for t in _REQUIRED_KEYS}
    for table, counts in ids.items():
        problems += [f'{table}: id {i} is duplicated' for i,
                     n in counts.items() if n > 1]

    for s in coco['subsets']:
        if s['dataset_id'] not in ids['datasets']:
            problems.append(
                f'subsets id={s["id"]}: dataset_id {s["dataset_id"]} does not exist')

    for i in coco['images']:
        where = f'images id={i["id"]}'
        if i['subset_id'] not in ids['subsets']:
            problems.append(
                f'{where}: subset_id {i["subset_id"]} does not exist')
        if i['climate_id'] is not None and i['climate_id'] not in ids['climates']:
            problems.append(
                f'{where}: climate_id {i["climate_id"]} does not exist')
        if i['time_of_day_id'] is not None and i['time_of_day_id'] not in ids['times_of_day']:
            problems.append(
                f'{where}: time_of_day_id {i["time_of_day_id"]} does not exist')
        problems += [
            f'{where}: condition_id {c} does not exist'
            for c in i['condition_ids'] if c not in ids['conditions']
        ]
        if i['hour'] is not None and not 0 <= i['hour'] <= 23:
            problems.append(f'{where}: hour {i["hour"]} outside 0-23')
        if not (isinstance(i['width'], int) and isinstance(i['height'], int)
                and i['width'] > 0 and i['height'] > 0):
            problems.append(f'{where}: width/height are not positive integers')
        if i['date'] is not None:
            try:
                Date.fromisoformat(i['date'])
            except (TypeError, ValueError):
                problems.append(
                    f'{where}: date {i["date"]!r} is not ISO YYYY-MM-DD')

    for a in coco['annotations']:
        where = f'annotations id={a["id"]}'
        if a['image_id'] not in ids['images']:
            problems.append(
                f'{where}: image_id {a["image_id"]} does not exist')
        if a['category_id'] not in ids['categories']:
            problems.append(
                f'{where}: category_id {a["category_id"]} does not exist')
        if a['spot_type_id'] not in ids['spot_types']:
            problems.append(
                f'{where}: spot_type_id {a["spot_type_id"]} does not exist')
        if a['truncated'] not in (0, 1):
            problems.append(
                f'{where}: truncated {a["truncated"]!r} is not 0/1')
    _fail_on(problems, IntegrityError)


def _check_geometry(coco: dict) -> None:
    problems: list[str] = []
    sizes = {i['id']: (i['width'], i['height']) for i in coco['images']}

    for a in coco['annotations']:
        where = f'annotations id={a["id"]}'
        seg, bbox, area = a['segmentation'], a['bbox'], a['area']
        poly = seg[0] if len(seg) == 1 else None
        shape_ok = (
            poly is not None and len(poly) >= 6 and len(poly) % 2 == 0
            and len(bbox) == 4
        )
        if not shape_ok:
            problems.append(
                f'{where}: segmentation must be 1 polygon with at least 3 points and bbox must have 4 values')
        elif not all(map(math.isfinite, [*poly, *bbox, area])):
            problems.append(f'{where}: has non-finite values')
        else:
            x, y, w, h = bbox
            width, height = sizes[a['image_id']]
            xs, ys = poly[0::2], poly[1::2]
            if w <= 0 or h <= 0 or area <= 0:
                problems.append(
                    f'{where}: bbox {bbox} or area {area} not positive')
            if (x < -_COORD_TOL or y < -_COORD_TOL
                    or x + w > width + _COORD_TOL or y + h > height + _COORD_TOL):
                problems.append(
                    f'{where}: bbox {bbox} falls outside the image {width}x{height}')
            drift = max(abs(min(xs) - x), abs(min(ys) - y),
                        abs(max(xs) - x - w), abs(max(ys) - y - h))
            if drift > _COORD_TOL:
                problems.append(
                    f'{where}: bbox does not match the polygon (off by {drift:.2f}px)')
    _fail_on(problems, GeometryError)


def _check_files(root: Path, images: list[dict]) -> None:
    from PIL import Image

    problems: list[str] = []

    for i in images:
        path = root / i['file_name']
        expected = (i['width'], i['height'])
        try:
            with Image.open(path) as im:
                if im.size != expected:
                    problems.append(
                        f'{i["file_name"]}: is {im.size}, annotated {expected}')
        except FileNotFoundError:
            problems.append(f'{i["file_name"]}: does not exist')
        except OSError as e:
            problems.append(f'{i["file_name"]}: unreadable ({e})')
    _fail_on(problems, ImageFileError)


@dataclass(frozen=True)
class Spot:
    id: int
    category_id: int  # as in the JSON: 1 empty, 2 occupied
    label: str
    bbox: tuple[float, float, float, float]  # x, y, width, height
    polygon: tuple[tuple[float, float], ...]
    area: float
    truncated: bool
    spot_type: str  # normal or cross

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def crop(self, image: Image.Image) -> Image.Image:
        x, y, w, h = self.bbox
        left, top = math.floor(x), math.floor(y)
        right = min(max(math.ceil(x + w), left + 1), image.width)
        bottom = min(max(math.ceil(y + h), top + 1), image.height)
        return image.crop((left, top, right, bottom))


@dataclass(frozen=True)
class Photo:
    id: int
    path: Path
    width: int
    height: int
    dataset: str
    parking: str
    camera: int
    climate: str | None
    time_of_day: str | None
    date: str | None
    hour: int | None
    conditions: tuple[str, ...]
    original_file: str  # path inside the source dataset
    # x, y, w, h cut from the original; None if uncropped
    crop: tuple[int, int, int, int] | None
    spots: tuple[Spot, ...]

    @property
    def empty_spots(self) -> tuple[Spot, ...]:
        return tuple(s for s in self.spots if s.label == 'empty')

    @property
    def occupied_spots(self) -> tuple[Spot, ...]:
        return tuple(s for s in self.spots if s.label == 'occupied')

    def image(self) -> Image.Image:
        from PIL import Image

        with Image.open(self.path) as im:
            return im.convert('RGB')

    def draw(self) -> Image.Image:
        """Image with the spots drawn: green empty, orange occupied."""
        from PIL import ImageDraw

        img = self.image()
        pen = ImageDraw.Draw(img)
        for s in self.spots:
            pen.polygon(s.polygon, outline='limegreen' if s.label ==
                        'empty' else 'orange')
        return img


class MetaPKLotMini:
    """Loads `annotations.json` and validates it in stages (SchemaError, IntegrityError,
    GeometryError, ImageFileError). `check_files=False` skips opening the ~5 000 images."""

    def __init__(self, root: str | Path, *, check_files: bool = True):
        self.root = Path(root)
        ann_path = self.root / 'annotations.json'
        if not ann_path.is_file():
            raise AnnotationsNotFoundError([f'{ann_path} does not exist'])

        try:
            with open(ann_path, encoding='utf-8') as f:
                coco = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SchemaError([f'{ann_path} is not valid JSON: {e}']) from e

        try:
            _check_schema(coco)
            _check_integrity(coco)
            _check_geometry(coco)
        except DatasetError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            # Right keys, wrong types (e.g. "hour": "9"): report it instead of crashing.
            raise SchemaError(
                [f'unexpected structure: {type(e).__name__}: {e}']) from e
        if check_files:
            _check_files(self.root, coco['images'])

        labels = {c['id']: c['name'] for c in coco['categories']}
        climates = {c['id']: c['name'] for c in coco['climates']}
        times = {t['id']: t['name'] for t in coco['times_of_day']}
        conditions = {c['id']: c['name'] for c in coco['conditions']}
        spot_types = {t['id']: t['name'] for t in coco['spot_types']}
        datasets = {d['id']: d['name'] for d in coco['datasets']}
        subsets = {s['id']: s for s in coco['subsets']}

        spots: dict[int, list[Spot]] = {}
        for a in coco['annotations']:
            seg = a['segmentation'][0]
            spots.setdefault(a['image_id'], []).append(Spot(
                id=a['id'],
                category_id=a['category_id'],
                label=labels[a['category_id']],
                bbox=tuple(a['bbox']),
                polygon=tuple(zip(seg[0::2], seg[1::2])),
                area=a['area'],
                truncated=bool(a['truncated']),
                spot_type=spot_types[a['spot_type_id']],
            ))

        self.datasets = tuple(datasets.values())
        self.parkings = tuple(dict.fromkeys(
            s['parking'] for s in subsets.values()))
        self.climates = tuple(climates.values())
        self.times_of_day = tuple(times.values())
        self.conditions = tuple(conditions.values())
        self._photos = tuple(
            Photo(
                id=i['id'],
                path=self.root / i['file_name'],
                width=i['width'],
                height=i['height'],
                dataset=datasets[subsets[i['subset_id']]['dataset_id']],
                parking=subsets[i['subset_id']]['parking'],
                camera=subsets[i['subset_id']]['camera'],
                climate=climates.get(i['climate_id']),
                time_of_day=times.get(i['time_of_day_id']),
                date=i['date'],
                hour=i['hour'],
                conditions=tuple(conditions[c] for c in i['condition_ids']),
                original_file=i['original']['file_name'],
                crop=tuple(i['original']['crop']
                           ) if i['original']['crop'] else None,
                spots=tuple(spots.get(i['id'], ())),
            )
            for i in coco['images']
        )
        self._by_id = {p.id: p for p in self._photos}

    def __len__(self) -> int:
        return len(self._photos)

    def __iter__(self):
        return iter(self._photos)

    def __repr__(self) -> str:
        return f'MetaPKLotMini(root={str(self.root)!r}, photos={len(self._photos)})'

    def photo(self, image_id: int) -> Photo:
        """By COCO image id (they start at 1). Raises KeyError if there is none."""
        return self._by_id[image_id]

    def photos(
        self,
        *,
        dataset: str | None = None,
        parking: str | None = None,
        camera: int | None = None,
        climate: str | None = None,
        time_of_day: str | None = None,
        date: str | None = None,
        condition: str | None = None,
    ) -> list[Photo]:
        """Filters by equality; `None` means no filter. `condition` requires the photo to
        have it. An unknown dataset, parking, climate, time of day or condition raises
        UnknownNameError; `camera` and `date` are open values and just match nothing."""
        bad = [
            f'{value!r} is not in {valid}'
            for value, valid in (
                (dataset, self.datasets),
                (parking, self.parkings),
                (climate, self.climates),
                (time_of_day, self.times_of_day),
                (condition, self.conditions),
            )
            if value is not None and value not in valid
        ]
        if bad:
            raise UnknownNameError('; '.join(bad))

        wanted = {
            'dataset': dataset, 'parking': parking, 'camera': camera,
            'climate': climate, 'time_of_day': time_of_day, 'date': date,
        }
        wanted = {k: v for k, v in wanted.items() if v is not None}
        return [
            p for p in self._photos
            if all(getattr(p, k) == v for k, v in wanted.items())
            and (condition is None or condition in p.conditions)
        ]


def split_photos(
    photos: Iterable[Photo], *, test_size: float = 0.2, seed: int = 99
) -> tuple[list[Photo], list[Photo]]:
    """Leak-free train/test split. Stratifies by (parking, camera) and keeps photos of
    the same day together, because they are near-duplicates. Test size is approximate,
    and a stratum with a single day goes entirely to train. Photos without a date
    (ACPDS, ACMPS) each count as their own group, so they can still leak."""
    if not 0 < test_size < 1:
        raise ValueError(f'test_size must be in (0, 1), got {test_size}')

    rng = random.Random(seed)
    strata: dict[tuple, dict] = {}
    for p in photos:
        strata.setdefault((p.parking, p.camera), {}).setdefault(
            p.date or p.id, []).append(p)

    train: list[Photo] = []
    test: list[Photo] = []
    for groups in strata.values():
        days = list(groups.values())
        rng.shuffle(days)
        goal = test_size * sum(len(d) for d in days)
        taken = 0
        n = 0
        while n < len(days) - 1 and taken < goal:
            test += days[n]
            taken += len(days[n])
            n += 1
        for d in days[n:]:
            train += d
    return train, test
