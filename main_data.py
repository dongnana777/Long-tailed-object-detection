# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_data import evaluate, lvis_evaluate, train_one_epoch
from models import build_model_data
import os
import cv2
import collections

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=True, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=3, type=float)#3
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='lvis')
    parser.add_argument("--lvis_path", default='', type=str)

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_backbone', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--sampling', default=True, action='store_true')#False
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument("--label_map", default=False, action="store_true")
    parser.add_argument('--lvis_version', default='1', help='lvis dataset version')
    parser.add_argument('--class_scores_paths',
                        default='./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/resnet50_lvis1/class_scores.pth')
    parser.add_argument('--ids_paths',
                        default='./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/resnet50_lvis1/ids.pth')
    parser.add_argument('--image_ids_paths',
                        default='./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/resnet50_lvis1/image_ids.pth')
    parser.add_argument('--ids_ratio_paths',
                        default='./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/ids_ratio.pth')
    parser.add_argument('--stage', default="generate_data", type=str)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model_data(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     import copy
        #     p_groups = copy.deepcopy(optimizer.param_groups)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for pg, pg_old in zip(optimizer.param_groups, p_groups):
        #         pg['lr'] = pg_old['lr']
        #         pg['initial_lr'] = pg_old['initial_lr']
        #     print(optimizer.param_groups)
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
        #     args.override_resumed_lr_drop = True
        #     if args.override_resumed_lr_drop:
        #         print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
        #         lr_scheduler.step_size = args.lr_drop
        #         lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        #     lr_scheduler.step(lr_scheduler.last_epoch)
        #     args.start_epoch = checkpoint['epoch'] + 1
        # # check the resumed model
        # if not args.eval:
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #     )

    if args.eval:
        if args.dataset_file !='lvis':
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        else:
            test_stats, coco_evaluator = lvis_evaluate(model, criterion, postprocessors,
                                                       data_loader_val, base_ds, device, args.output_dir,
                                                       args.label_map)

        return

    if args.sampling:
        num_classes = 1230 + 1
        # first
        test_stats, coco_evaluator, class_scores, ids, image_ids = lvis_evaluate(model, criterion, postprocessors,
                                                                        data_loader_val, base_ds, device,
                                                                        args.output_dir, args.label_map, num_classes)

        utils.save_on_master(class_scores, output_dir / "class_scores.pth")
        utils.save_on_master(ids, output_dir / "ids.pth")
        utils.save_on_master(image_ids, output_dir / "image_ids.pth")
    else:
        # second
        ids = torch.load(args.ids_paths, map_location='cpu')
        class_scores = torch.load(args.class_scores_paths, map_location='cpu')
        image_ids = torch.load(args.image_ids_paths, map_location='cpu')

        for m in [500]:
            exemplar_id_sets = dict.fromkeys(range(1, 1204), [])
            for (k, v), (k1, v1), (k2, v2) in zip(class_scores.items(), ids.items(), image_ids.items()):
                v = torch.stack(v)
                v1 = torch.stack(v1)
                v2 = torch.stack(v2)
                classes_scores, classes_indices = torch.sort(v, dim=0, descending=True)
                satisfied_idx = classes_indices[:m]
                satisfied_ids = v1[satisfied_idx]

                ids_temp = []
                image_ids_temp = []
                score_temp = []
                for ind in classes_indices:
                    if v2[ind] not in image_ids_temp:
                        if v[ind] > 0.5:
                            ids_temp.append(v1[ind])
                            image_ids_temp.append(v2[ind])
                            score_temp.append(v[ind])
                ids_temp = ids_temp + list(satisfied_ids)

                ids_temp_2 = []
                for i in ids_temp:
                    if i not in ids_temp_2:
                        ids_temp_2.append(i)

                ids_temp = torch.tensor(ids_temp_2)
                # if ids_temp.shape[0] < m:
                #     ids_temp = ids_temp.repeat(1, math.ceil(m/ids_temp.shape[0])).squeeze(0)
                #     exemplar_id_sets[k] = ids_temp[:m]
                # else:
                #     exemplar_id_sets[k] = ids_temp[:m]
                exemplar_id_sets[k] = ids_temp[:m]

            # for (k, v), (k1, v1) in zip(class_scores.items(), ids.items()):
            #     v = torch.stack(v)
            #     v1 = torch.stack(v1)
            #     classes_scores, classes_indices = torch.sort(v, dim=0, descending=True)
            #     satisfied_idx = classes_indices[:m]
            #     satisfied_ids = v1[satisfied_idx]
            #     exemplar_id_sets[k] = satisfied_ids

            utils.save_on_master(exemplar_id_sets, output_dir / "exemplar_id_sets_{}.pth".format(m))

        # confidence
        # ids = torch.load(args.ids_paths, map_location='cpu')
        # class_scores = torch.load(args.class_scores_paths, map_location='cpu')
        # ids_ratio = torch.load(args.ids_ratio_paths, map_location='cpu')
        #
        # exemplar_id_sets = dict.fromkeys(range(1, 1204), [])
        # for (k, v), (k1, v1), (k2, v2) in zip(class_scores.items(), ids.items(), ids_ratio.items()):
        #     if len(v) > v2:
        #         v = torch.stack(v)
        #         v1 = torch.stack(v1)
        #         classes_scores, classes_indices = torch.sort(v, dim=0, descending=True)
        #         satisfied_idx = classes_indices[:v2]
        #         satisfied_ids = v1[satisfied_idx]
        #         exemplar_id_sets[k] = satisfied_ids
        #     else:
        #         exemplar_id_sets[k] = ids[k]
        #
        #     utils.save_on_master(exemplar_id_sets, output_dir / "exemplar_id_sets_ratio.pth")

        # NCM
        # class_prototypes = {}
        # for k, v in class_features.items():
        #     if len(v) != 0:
        #         v_mean = torch.stack(v, dim=0).mean(0)
        #         class_prototypes[k] = v_mean
        # utils.save_on_master(class_prototypes, output_dir / "class_prototypes.pth")

        # contruct exemplar set
        # for m in [5, 10, 50, 100, 200]:
        #     exemplar_id_sets = dict.fromkeys(range(1, 1204), [])
        #     for (k, v), (k1, v1) in zip(class_features.items(), ids.items()):
        #         if len(v) > m:
        #             temp_list = []
        #             temp_list_id = []
        #             for class_feature in class_features[k]:
        #                 temp_list.append(class_feature.numpy())
        #             for id in ids[k1]:
        #                 temp_list_id.append(id.numpy())
        #             id_temp = np.array(temp_list_id)
        #             features = np.array(temp_list)
        #             exemplar_id_set = []
        #             exemplar_features = []  # list of Variables of shape (feature_size,)
        #             for j in range(m):
        #                 S = np.sum(exemplar_features, axis=0)
        #                 phi = features
        #                 mu = class_prototypes[k].unsqueeze(0).numpy()
        #                 mu = mu / np.linalg.norm(mu)
        #                 mu_p = 1.0 / (j + 1) * (phi + S)
        #                 mu_p = mu_p / np.linalg.norm(mu_p)
        #                 i = np.argmax(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
        #
        #                 # if image_ids[k][i] not in exemplar_id_set:
        #                 exemplar_id_set.append(id_temp[i])
        #                 exemplar_features.append(features[i])
        #                 features = np.delete(features, i, axis=0)
        #                 id_temp = np.delete(id_temp, i, axis=0)
        #
        #             exemplar_id_sets[k] = exemplar_id_set
        #         else:
        #             exemplar_id_sets[k] = ids[k]
        #
        #      utils.save_on_master(exemplar_id_sets, output_dir / "exemplar_id_sets_{}.pth".format(m))

        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
