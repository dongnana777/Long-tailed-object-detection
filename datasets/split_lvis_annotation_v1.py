import argparse
import json
import os
from datasets.lvis_classes import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default="./lvis_v1_train.json",
                        help='path to the annotation file')
    parser.add_argument('--save-dir', type=str,
                        default="",
                        help='path to the save directory')
    args = parser.parse_args()
    return args

def split_annotation(args):
    with open(args.data) as fp:
        ann_train = json.load(fp)

    ann_s = {
        'info': ann_train['info'],
        # 'images': ann_train['images'],
        'categories': ann_train['categories'],
        'licenses': ann_train['licenses'],
    }
    ids = common_30_100_v1
    name = '30_100'
    ann_s['annotations'] = [ann for ann in ann_train['annotations'] if ann['category_id'] in ids]
    img_ids = set([ann['image_id'] for ann in ann_s['annotations']])
    new_images = [img for img in ann_train['images'] if img['id'] in img_ids]
    ann_s['images'] = new_images

    save_path = os.path.join(
        args.save_dir, 'lvis_v1_train_{}.json'.format(name))
    print('Saving {} annotations to {}.'.format(name, save_path))
    with open(save_path, 'w') as fp:
        json.dump(ann_s, fp)


if __name__ == '__main__':
    args = parse_args()
    split_annotation(args)
