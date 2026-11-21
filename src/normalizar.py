"""Normalización de anotaciones de MetaPKLot (spots) a un COCO único y limpio.

Uso en el notebook (sustituye a las celdas que rellenan `mini`):

    from normalizar import construir_coco_mini, validar_coco, copiar_imagenes
    mini, recortes = construir_coco_mini(spots_files, tabla_muestreada)
    validar_coco(mini)
    copiar_imagenes(mini, recortes, CONFIG['dataset']['source']['path'],
                    CONFIG['dataset']['mini']['path'])

Decisiones (ver el README del dataset):
  * Categorías COCO 1-based: 1 = empty, 2 = occupied (0 queda libre para el
    fondo, que es lo que esperan torchvision/Faster R-CNN). Para YOLO: clase = id - 1.
  * Ids secuenciales (imágenes 1..N, anotaciones 1..M); el id de MetaPKLot se
    conserva en `original_id`.
  * Cada imagen se recorta a su `annotationsRectangle`, porque fuera de esa
    región hay plazas sin anotar que el detector aprendería como fondo.
    Las coordenadas se desplazan al origen del recorte.
  * Geometría en float con 2 decimales. `bbox` y `area` se recalculan a partir
    del polígono (ya recortado), así PKLot (enteros) y CNR (decimales) quedan
    con el mismo criterio.
  * Plazas parcialmente fuera del recorte: se recorta el polígono y se marca
    `truncated = 1`; si queda menos de `min_visible` de su área, se descarta.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections import Counter

CATEGORIAS = [
    {'id': 1, 'name': 'empty', 'supercategory': 'parking_spot'},
    {'id': 2, 'name': 'occupied', 'supercategory': 'parking_spot'},
]
MAPA_CATEGORIAS = {0: 1, 1: 2}  # MetaPKLot -> mini
NOMBRE_DATASET = {'pklot': 'PKLot', 'cnr': 'CNRPark-EXT'}
DEC = 2


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #
def _area_poligono(pts):
    """Área por la fórmula del lazo (shoelace). pts = [(x, y), ...]."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n))
    return abs(s) / 2.0


def _recortar_poligono(pts, x0, y0, x1, y1):
    """Sutherland–Hodgman contra el rectángulo [x0, x1] x [y0, y1]."""
    def recortar(puntos, dentro, interseccion):
        salida = []
        for i, actual in enumerate(puntos):
            previo = puntos[i - 1]
            if dentro(actual):
                if not dentro(previo):
                    salida.append(interseccion(previo, actual))
                salida.append(actual)
            elif dentro(previo):
                salida.append(interseccion(previo, actual))
        return salida

    def corte_x(xc):
        return lambda p, q: (xc, p[1] + (q[1] - p[1]) * (xc - p[0]) / (q[0] - p[0]))

    def corte_y(yc):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (yc - p[1]) / (q[1] - p[1]), yc)

    for dentro, inter in (
        (lambda p: p[0] >= x0, corte_x(x0)),
        (lambda p: p[0] <= x1, corte_x(x1)),
        (lambda p: p[1] >= y0, corte_y(y0)),
        (lambda p: p[1] <= y1, corte_y(y1)),
    ):
        if not pts:
            break
        pts = recortar(pts, dentro, inter)
    return pts


def _a_pares(flat):
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def normalizar_plaza(plaza, rect, min_visible=0.5):
    """Devuelve la plaza normalizada en coordenadas del recorte, o None si se descarta.

    rect = (x, y, w, h) de annotationsRectangle en coordenadas de la imagen original.
    """
    rx, ry, rw, rh = (float(v) for v in rect)
    pts = _a_pares(plaza['segmentation'][0])
    area_orig = _area_poligono(pts)
    if area_orig <= 0:
        return None

    recortado = _recortar_poligono(pts, rx, ry, rx + rw, ry + rh)
    area = _area_poligono(recortado)
    if area < min_visible * area_orig:
        return None
    truncada = int(area < area_orig - 1e-6)

    # Desplazar al origen del recorte y redondear
    pts = [(round(x - rx, DEC), round(y - ry, DEC)) for x, y in recortado]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bbox = [min(xs), min(ys), round(max(xs) - min(xs), DEC), round(max(ys) - min(ys), DEC)]
    if bbox[2] < 1 or bbox[3] < 1:
        return None

    (cx, cy), (w, h), ang = plaza['best_rotated_rect']
    return {
        'category_id': MAPA_CATEGORIAS[plaza['category_id']],
        'segmentation': [[c for p in pts for c in p]],
        'area': round(area, DEC),
        'bbox': bbox,
        'iscrowd': 0,
        'truncated': truncada,
        # Sigue el convenio de cv2.minAreaRect; solo es exacto si truncated == 0
        'best_rotated_rect': [[round(float(cx) - rx, DEC), round(float(cy) - ry, DEC)],
                              [round(float(w), DEC), round(float(h), DEC)],
                              round(float(ang), DEC)],
        'original_id': plaza['id'],
    }


# --------------------------------------------------------------------------- #
# Construcción del COCO mini
# --------------------------------------------------------------------------- #
def _leer_coco(path):
    with tarfile.open(path, 'r:xz') as tar:
        member = next(m for m in tar if m.name.endswith('.json'))
        return json.load(tar.extractfile(member))


def construir_coco_mini(spots_files, tabla_muestreada, min_visible=0.5,
                        descripcion='Meta-PKLot-mini: 5,000 imágenes de PKLot y CNRPark-EXT'):
    """Devuelve (coco_mini, recortes) donde recortes = {image_id_nuevo: (x, y, w, h)}."""
    meta = tabla_muestreada.set_index('image_id')
    elegidos = set(meta.index)

    mini = {
        'info': {'description': descripcion, 'version': '1.0',
                 'source': 'MetaPKLot (annotations/original/spots)'},
        'licenses': [],
        'categories': CATEGORIAS,
        'images': [],
        'annotations': [],
    }
    recortes, descartadas = {}, Counter()
    img_id, ann_id = 0, 0

    for path in spots_files:
        if f'{os.sep}PLds{os.sep}' in path or '/PLds/' in path:
            continue
        coco = _leer_coco(path)
        plazas_por_imagen = {}
        for a in coco['annotations']:
            if a['image_id'] in elegidos:
                plazas_por_imagen.setdefault(a['image_id'], []).append(a)

        for img in sorted(coco['images'], key=lambda i: i['id']):
            if img['id'] not in elegidos:
                continue
            fila = meta.loc[img['id']]
            rect = img.get('annotationsRectangle') or [0, 0, img['width'], img['height']]
            # Asegurar que el recorte no se sale de la imagen
            x, y = max(0, int(rect[0])), max(0, int(rect[1]))
            w = min(int(rect[2]), img['width'] - x)
            h = min(int(rect[3]), img['height'] - y)

            img_id += 1
            recortes[img_id] = (x, y, w, h)
            n_libres = n_ocupadas = 0
            for plaza in plazas_por_imagen.get(img['id'], []):
                norm = normalizar_plaza(plaza, (x, y, w, h), min_visible)
                if norm is None:
                    descartadas[img['subset']] += 1
                    continue
                ann_id += 1
                mini['annotations'].append({'id': ann_id, 'image_id': img_id, **norm})
                n_libres += norm['category_id'] == 1
                n_ocupadas += norm['category_id'] == 2

            mini['images'].append({
                'id': img_id,
                'file_name': f"images/{img['id']}.jpg",
                'width': w,
                'height': h,
                'original_id': img['id'],
                'original_file_name': img['file_name'],
                'original_size': [img['width'], img['height']],
                'crop': [x, y, w, h],
                'dataset': NOMBRE_DATASET[img['dataset']],
                'subset': img['subset'],
                'fecha': fila['fecha'],
                'hora': int(fila['hora']),
                'franja_horaria': fila['franja_horaria'],
                'clima': fila['clima'],
                'plazas_libres': int(n_libres),
                'plazas_ocupadas': int(n_ocupadas),
            })

    if descartadas:
        print('Plazas descartadas por quedar fuera del recorte:', dict(descartadas))
    return mini, recortes


# --------------------------------------------------------------------------- #
# Validación
# --------------------------------------------------------------------------- #
def validar_coco(coco, tol=0.01):
    """Lanza AssertionError si algo no cuadra; imprime un resumen si todo va bien."""
    imgs = {i['id']: i for i in coco['images']}
    cats = {c['id'] for c in coco['categories']}
    assert len(imgs) == len(coco['images']), 'ids de imagen duplicados'
    assert len({a['id'] for a in coco['annotations']}) == len(coco['annotations']), \
        'ids de anotación duplicados'
    assert len({i['file_name'] for i in coco['images']}) == len(imgs), 'file_name duplicados'

    por_imagen = Counter()
    for a in coco['annotations']:
        assert a['image_id'] in imgs, f"anotación {a['id']} sin imagen"
        assert a['category_id'] in cats, f"categoría inválida en {a['id']}"
        im = imgs[a['image_id']]
        x, y, w, h = a['bbox']
        assert w > 0 and h > 0, f"bbox degenerada en {a['id']}"
        assert x >= -tol and y >= -tol and x + w <= im['width'] + tol \
            and y + h <= im['height'] + tol, f"bbox fuera de la imagen en {a['id']}"
        seg = a['segmentation'][0]
        assert len(seg) >= 6 and len(seg) % 2 == 0, f"polígono inválido en {a['id']}"
        assert 0 < a['area'] <= w * h + tol, f"área incoherente en {a['id']}"
        por_imagen[a['image_id']] += 1

    sin_plazas = [i for i in imgs if por_imagen[i] == 0]
    assert not sin_plazas, f'{len(sin_plazas)} imágenes sin plazas'
    for im in coco['images']:
        assert im['plazas_libres'] + im['plazas_ocupadas'] == por_imagen[im['id']], \
            f"recuento incoherente en imagen {im['id']}"

    clases = Counter(a['category_id'] for a in coco['annotations'])
    nombres = {c['id']: c['name'] for c in coco['categories']}
    trunc = sum(a.get('truncated', 0) for a in coco['annotations'])
    print(f"OK · {len(imgs)} imágenes · {len(coco['annotations'])} plazas · "
          f"{ {nombres[k]: v for k, v in sorted(clases.items())} } · truncadas: {trunc}")


# --------------------------------------------------------------------------- #
# Imágenes
# --------------------------------------------------------------------------- #
def copiar_imagenes(coco, recortes, origen, destino, calidad=95):
    """Guarda cada imagen recortada a su annotationsRectangle en destino/images/."""
    from PIL import Image

    os.makedirs(os.path.join(destino, 'images'), exist_ok=True)
    faltan = []
    for im in coco['images']:
        src = os.path.join(origen, im['original_file_name'])
        if not os.path.exists(src):
            faltan.append(src)
            continue
        x, y, w, h = recortes[im['id']]
        with Image.open(src) as foto:
            assert foto.size == tuple(im['original_size']), f'tamaño inesperado: {src}'
            foto.crop((x, y, x + w, y + h)).save(os.path.join(destino, im['file_name']),
                                                  quality=calidad)
    if faltan:
        raise FileNotFoundError(f'{len(faltan)} imágenes no encontradas, p. ej. {faltan[0]}')
    print('Imágenes guardadas:', len(coco['images']))


def exportar_yolo(coco, destino):
    """Etiquetas YOLO (detección) en destino/labels/: clase = category_id - 1."""
    os.makedirs(os.path.join(destino, 'labels'), exist_ok=True)
    lineas = {i['id']: [] for i in coco['images']}
    imgs = {i['id']: i for i in coco['images']}
    for a in coco['annotations']:
        im = imgs[a['image_id']]
        x, y, w, h = a['bbox']
        lineas[a['image_id']].append(
            f"{a['category_id'] - 1} {(x + w / 2) / im['width']:.6f} "
            f"{(y + h / 2) / im['height']:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
    for iid, ls in lineas.items():
        nombre = os.path.splitext(os.path.basename(imgs[iid]['file_name']))[0]
        with open(os.path.join(destino, 'labels', nombre + '.txt'), 'w') as f:
            f.write('\n'.join(ls) + '\n')
