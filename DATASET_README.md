---
pretty_name: Meta-PKLot-mini
license: other
language:
  - es
task_categories:
  - object-detection
tags:
  - computer-vision
  - parking
  - parking-lot
  - occupancy-recognition
  - coco
source_datasets:
  - DSBD-Research/MetaPKLot-Dataset
---

> TODO: Es plantilla de referncia, habra que cambiar

# Meta-PKLot-mini

Subconjunto reducido de [MetaPKLot](https://huggingface.co/datasets/DSBD-Research/MetaPKLot-Dataset), preparado para detectar plazas de aparcamiento libres y ocupadas.

## Origen


[MetaPKLot](https://huggingface.co/datasets/DSBD-Research/MetaPKLot-Dataset) unifica y reanota tres datasets de aparcamientos: PKLot, CNRPark-EXT y PLds. Aquí solo se usan las anotaciones de plazas (*spots*) y sus imágenes.

## Contenido

> Completar al terminar el preprocesamiento.

| Propiedad              | Valor         |
| ---------------------- | ------------- |
| Subconjuntos de origen | _por definir_ |
| Imágenes               | _por definir_ |
| Plazas anotadas        | _por definir_ |
| Particiones            | _por definir_ |

### Estructura

```text
Meta-PKLot-mini/
├── images/
└── annotations/
```

> Ajustar al resultado real.

### Formato de las anotaciones

JSON tipo COCO con las secciones `info`, `licenses`, `images`, `categories` y `annotations`.

| `category_id` | Clase      |
| ------------- | ---------- |
| 0             | `empty`    |
| 1             | `occupied` |

- `bbox`: `[x, y, ancho, alto]`, con `(x, y)` en la esquina superior izquierda.
- `annotationsRectangle`: región con anotación completa. Lo que queda fuera no es un negativo.
- `file_name`: ruta POSIX relativa a la raíz del dataset.

## Procesamiento aplicado

> Completar: criterios de selección, filtros y conversiones respecto a MetaPKLot.

## Uso

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='StafLoker/Meta-PKLot-mini',
    repo_type='dataset',
    local_dir='Meta-PKLot-mini',
)
```

## Autores

Anthony Manzueta Difo, Stefan Oshchypok, Marina Ruiz Izu.
