import os
import json
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw

def read_json(filename: str):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann
data = read_json("../../data/medical/ufo/output.json")

img_lists = glob('../../data/medical/img/test/*.jpg')
img_lists = [i.split('/')[-1] for i in img_lists]

def save_vis_to_img(save_dir = None, img_lists: list = None) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ori_dict = {
        "Horizontal": "ho",
        "Vertical": "ve",
        "Irregular": "ir"
    }

    lag_dict = {
        None: '0',
        'others': '1',
        'ko': '2',
        'en': '3',
        'ko, others': '4',
        'ko, en': '5',
    }

    tag_dict = {
        "occlusion": "occ",
        "stamp": "stamp",
        "masked": "mask",
        "inferred": "infer"
    }

    for i in range(len(img_lists)):
        img_json = [[k, v] for k, v in data['images'].items() if k == img_lists[i]]
        img_path = img_json[0][0]
        img = Image.open(os.path.join('../../data/medical/img/test', img_path)).convert("RGB")
        draw = ImageDraw.Draw(img)

         # All of the prepared dataset consists of words. Not a character.
        for obj_k, obj_v in img_json[0][1]['words'].items():
            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]
            
            draw.polygon(pts, outline=(255, 0, 0))
            img.save(os.path.join(save_dir, img_path))

save_vis_to_img("vis_res", img_lists)