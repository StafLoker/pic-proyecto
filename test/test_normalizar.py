import io, json, os, sys, tarfile
import pandas as pd
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from normalizar import construir_coco_mini, validar_coco, copiar_imagenes, exportar_yolo

import tempfile
TMP = tempfile.mkdtemp()
base = os.path.join(TMP, 'MetaPKLot')
def tar_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj).encode()
    with tarfile.open(path, 'w:xz') as tar:
        ti = tarfile.TarInfo('x.json'); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))
def jpg(rel, w, h):
    p = os.path.join(base, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.new('RGB', (w, h), 'gray').save(p)

climates = [{'id': i, 'name': n} for i, n in enumerate(['Overcast','Cloudy','Sunny','Rainy','Snowy','Clearnight'])]
pk_img = {"id": 11000002330, "file_name": "PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.jpg",
  "width": 1280, "height": 720, "annotationsRectangle": [0, 140, 1280, 580],
  "date": [2012,9,11], "time": [15,16,58], "climate": 2, "dataset": "pklot", "subset": "pucpr"}
pk_anns = [
  {"id": 11000000000, "image_id": 11000002330, "category_id": 1,
   "segmentation": [[278,230,290,186,324,185,308,230]], "bbox": [278,185,46,45], "area": 1417.0,
   "iscrowd": 0, "car_id": 1, "best_rotated_rect": [[299.03,205.51],[32.27,58.68],14.93]},
  {"id": 11000000001, "image_id": 11000002330, "category_id": 0,
   "segmentation": [[400,120,440,120,440,170,400,170]], "bbox": [400,120,40,50], "area": 2000.0,
   "iscrowd": 0, "car_id": -1, "best_rotated_rect": [[420,145],[40,50],0]},
  {"id": 11000000002, "image_id": 11000002330, "category_id": 0,
   "segmentation": [[500,100,540,100,540,145,500,145]], "bbox": [500,100,40,45], "area": 1800.0,
   "iscrowd": 0, "car_id": -1, "best_rotated_rect": [[520,122.5],[40,45],0]},
]
cnr_img = {"id": 21000000403, "file_name": "CNRPark-EXT/camera1/SUNNY/2015-11-12/2015-11-12_07_09_00.jpg",
  "width": 1000, "height": 750, "annotationsRectangle": [78, 158, 922, 592],
  "date": [2015,11,12], "time": [7,9,0], "climate": 2, "dataset": "cnr", "subset": "camera1"}
cnr_anns = [{"id": 21000000000, "image_id": 21000000403, "category_id": 0,
  "segmentation": [[391.4,328.6,435.0,357.4,347.0,419.5,296.0,392.4]], "area": 5544.0,
  "bbox": [296.0,328.6,139.0,90.9], "iscrowd": 0, "car_id": -1,
  "best_rotated_rect": [[366.25,376.02],[50.88,135.0],56.23]}]
cats = [{'id':0,'name':'empty'},{'id':1,'name':'occupied'}]
f1 = f'{base}/annotations/original/spots/PKLot/pucpr_spots.tar.xz'
f2 = f'{base}/annotations/original/spots/CNRPark-EXT/cnr-camera-1_spots.tar.xz'
f3 = f'{base}/annotations/original/spots/PLds/isshk_spots.tar.xz'
tar_json(f1, {'categories':cats,'climates':climates,'images':[pk_img],'annotations':pk_anns,'info':{}})
tar_json(f2, {'categories':cats,'climates':climates,'images':[cnr_img],'annotations':cnr_anns,'info':{}})
tar_json(f3, {'categories':cats,'climates':climates,'images':[],'annotations':[],'info':{}})
jpg(pk_img['file_name'], 1280, 720); jpg(cnr_img['file_name'], 1000, 750)

tabla = pd.DataFrame([
  dict(image_id=11000002330, dataset='pklot', subset='pucpr', fecha='2012-09-11', hora=15, clima='Sunny', franja_horaria='mediodia'),
  dict(image_id=21000000403, dataset='cnr', subset='camera1', fecha='2015-11-12', hora=7, clima='Sunny', franja_horaria='mañana')])

mini, rec = construir_coco_mini(sorted([f1, f2, f3]), tabla)
validar_coco(mini)
copiar_imagenes(mini, rec, base, os.path.join(TMP, 'mini'))
exportar_yolo(mini, os.path.join(TMP, 'mini'))
print(json.dumps(mini['images'][0], ensure_ascii=False))
for a in mini['annotations']: print(a)
for f in sorted(os.listdir(os.path.join(TMP, 'mini', 'images'))): print(f, Image.open(os.path.join(TMP, 'mini', 'images', f)).size)
for f in sorted(os.listdir(os.path.join(TMP, 'mini', 'labels'))): print(f, open(os.path.join(TMP, 'mini', 'labels', f)).read().strip())

# Comprobaciones automáticas
assert len(mini['images']) == 2 and len(mini['annotations']) == 3
assert {c['id'] for c in mini['categories']} == {1, 2}
pucpr = next(i for i in mini['images'] if i['subset'] == 'pucpr')
assert (pucpr['width'], pucpr['height']) == (1280, 580)
a = next(a for a in mini['annotations'] if a['original_id'] == 11000000000)
assert a['bbox'] == [278.0, 45.0, 46.0, 45.0] and a['category_id'] == 2
assert next(a for a in mini['annotations'] if a['original_id'] == 11000000001)['truncated'] == 1
assert not any(a['original_id'] == 11000000002 for a in mini['annotations'])
print('\nTODOS LOS TESTS PASAN')
