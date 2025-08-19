import argparse
import json
import os
import torch
from datasets.lvis_classes import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default="./lvis_v0.5_train.json",
                        help='path to the annotation file')
    parser.add_argument('--save-dir', type=str,
                        default="",
                        help='path to the save directory')
    parser.add_argument('--exemplar_id_sets_freq_paths',
                        default='')
    parser.add_argument('--exemplar_id_sets_common_paths',
                        default='')

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

    classes_exemplars_ids = []
    if args.exemplar_id_sets_freq_paths:
        exemplar_id_sets = torch.load(args.exemplar_id_sets_freq_paths, map_location='cpu')
        for k, v in exemplar_id_sets.items():
            if k in freq_classes_v5:
                v = list(map(int, v))[:200]
                classes_exemplars_ids.extend(v)
    if args.exemplar_id_sets_common_paths:
        exemplar_id_sets = torch.load(args.exemplar_id_sets_common_paths, map_location='cpu')
        for k, v in exemplar_id_sets.items():
            if k in common_30_100_v5:
                v = list(map(int, v))[:200]
                classes_exemplars_ids.extend(v)
    if args.exemplar_id_sets_common_paths:
        exemplar_id_sets = torch.load(args.exemplar_id_sets_common_paths, map_location='cpu')
        for k, v in exemplar_id_sets.items():
            if k in common_10_30_v5:
                v = list(map(int, v))[:100]
                classes_exemplars_ids.extend(v)

    ann = []
    for s, name in [('r', 'rare')]:  # ('f', 'freq'), ('c', 'common')
        ids = [cat['id'] for cat in ann_train['categories'] if cat['frequency'] == s]
        ann.extend([ann for ann in ann_train['annotations'] if ann['category_id'] in ids])

    ann.extend([ann for ann in ann_train['annotations'] if ann['id'] in list(classes_exemplars_ids)])
    img_ids = set([ann['image_id'] for ann in ann])
    ann_s['annotations'] = ann
    new_images = [img for img in ann_train['images'] if img['id'] in img_ids]
    ann_s['images'] = new_images

    name = 'freq_common_rare_lt_set'
    save_path = os.path.join(args.save_dir, 'lvis_v0.5_train_{}.json'.format(name))
    print('Saving {} annotations to {}.'.format(name, save_path))
    with open(save_path, 'w') as fp:
        json.dump(ann_s, fp)


if __name__ == '__main__':
    args = parse_args()
    split_annotation(args)
