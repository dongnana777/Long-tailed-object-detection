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
from engine_local0 import evaluate, lvis_evaluate, train_one_epoch0
from engine_local1 import evaluate, lvis_evaluate, train_one_epoch1
from models import build_model0, build_model1, build_model_teacher
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
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs_l0', default=1, type=int)
    parser.add_argument('--epochs_l1', default=2, type=int)
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
    parser.add_argument('--backbone', default='resnet50', type=str,
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
    parser.add_argument('--cls_dis_shared_loss_coef', default=0.1, type=float)#0.1
    parser.add_argument('--feature_loss_coef', default=1, type=float)#5

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
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument("--label_map", default=False, action="store_true")
    parser.add_argument('--freeze_weight', default=True, help='whether to freeze the weight')
    parser.add_argument('--lvis_version', default='1', help='lvis dataset version')
    parser.add_argument('--stage', default="FT and KD", type=str)
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

    model_l0, criterion_l0, postprocessors = build_model0(args)
    model_l1, criterion_l1, postprocessors = build_model1(args)
    model_teacher, criterion_teacher, postprocessors = build_model_teacher(args)

    model_l0.to(device)
    model_l1.to(device)
    model_teacher.to(device)
    model_without_ddp_l0 = model_l0
    model_without_ddp_l1 = model_l1
    model_without_ddp_teacher = model_teacher

    n_parameters = sum(p.numel() for p in model_l0.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    n_parameters = sum(p.numel() for p in model_l1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    n_parameters = sum(p.numel() for p in model_teacher.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.freeze_weight:
        for n, p in model_without_ddp_teacher.named_parameters():
            p.requires_grad = False

        for n, p in model_without_ddp_l0.named_parameters():
            p.requires_grad = False
        for n, p in model_without_ddp_l0.named_parameters():
            if 'class_embed' in n or 'input_proj' in n:
                p.requires_grad = True
            else:
                pass

        for n, p in model_without_ddp_l1.named_parameters():
            p.requires_grad = False
        for n, p in model_without_ddp_l1.named_parameters():
            if 'class_embed' in n or 'input_proj' in n:
                p.requires_grad = True
            else:
                pass

    for n, p in model_without_ddp_teacher.named_parameters():
        if p.requires_grad:
            print(n)
    for n, p in model_without_ddp_l0.named_parameters():
        if p.requires_grad:
            print(n)
    for n, p in model_without_ddp_l1.named_parameters():
        if p.requires_grad:
            print(n)

    dataset_train_l0, dataset_train_l1 = build_dataset(image_set='train', args=args)
    dataset_val_l0, dataset_val_l1 = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train_l0 = samplers.NodeDistributedSampler(dataset_train_l0)
            sampler_val_l0 = samplers.NodeDistributedSampler(dataset_val_l0, shuffle=False)
        else:
            sampler_train_l0 = samplers.DistributedRepeatFactorReSampler_0(dataset_train_l0)
            sampler_val_l0 = samplers.DistributedSampler(dataset_val_l0, shuffle=False)
    else:
        sampler_train_l0 = torch.utils.data.RandomSampler(dataset_train_l0)
        sampler_val_l0 = torch.utils.data.SequentialSampler(dataset_val_l0)

    batch_sampler_train_l0 = torch.utils.data.BatchSampler(sampler_train_l0, args.batch_size, drop_last=True)
    # batch_sampler_train_l0 = samplers.AspectRatioGroupedBatchSampler(sampler_train_l0, args.batch_size, aspect_grouping=[1], drop_last=True)

    data_loader_train_l0 = DataLoader(dataset_train_l0, batch_sampler=batch_sampler_train_l0,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val_l0 = DataLoader(dataset_val_l0, args.batch_size, sampler=sampler_val_l0,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.distributed:
        if args.cache_mode:
            sampler_train_l1 = samplers.NodeDistributedSampler(dataset_train_l1)
            sampler_val_l1 = samplers.NodeDistributedSampler(dataset_val_l1, shuffle=False)
        else:
            sampler_train_l1 = samplers.DistributedRepeatFactorReSampler_1(dataset_train_l1)
            sampler_val_l1 = samplers.DistributedSampler(dataset_val_l1, shuffle=False)
    else:
        sampler_train_l1 = torch.utils.data.RandomSampler(dataset_train_l1)
        sampler_val_l1 = torch.utils.data.SequentialSampler(dataset_val_l1)

    batch_sampler_train_l1 = torch.utils.data.BatchSampler(sampler_train_l1, args.batch_size, drop_last=True)
    # batch_sampler_train_l1 = samplers.AspectRatioGroupedBatchSampler(sampler_train_l1, args.batch_size, aspect_grouping=[1], drop_last=True)

    data_loader_train_l1 = DataLoader(dataset_train_l1, batch_sampler=batch_sampler_train_l1,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val_l1 = DataLoader(dataset_val_l1, args.batch_size, sampler=sampler_val_l1,
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
    for n, p in model_without_ddp_l0.named_parameters():
        print(n)
    for n, p in model_without_ddp_l1.named_parameters():
        print(n)
    for n, p in model_without_ddp_teacher.named_parameters():
        print(n)

    param_dicts_l0 = [
        {
            "params":
                [p for n, p in model_without_ddp_l0.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr*0.1,
        },
        {
            "params": [p for n, p in model_without_ddp_l0.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone*0.1,
        },
        {
            "params": [p for n, p in model_without_ddp_l0.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult*0.1,
        }
    ]
    param_dicts_l1 = [
        {
            "params":
                [p for n, p in model_without_ddp_l1.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp_l1.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone*0.1,
        },
        {
            "params": [p for n, p in model_without_ddp_l1.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult*0.1,
        }
    ]


    if args.sgd:
        optimizer_l0 = torch.optim.SGD(param_dicts_l0, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer_l1 = torch.optim.SGD(param_dicts_l1, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer_l0 = torch.optim.AdamW(param_dicts_l0, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_l1 = torch.optim.AdamW(param_dicts_l1, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler_l0 = torch.optim.lr_scheduler.StepLR(optimizer_l0, args.lr_drop)
    lr_scheduler_l1 = torch.optim.lr_scheduler.StepLR(optimizer_l1, args.lr_drop)

    if args.distributed:
        model_l0 = torch.nn.parallel.DistributedDataParallel(model_l0, device_ids=[args.gpu])
        model_without_ddp_l0 = model_l0.module
        model_l1 = torch.nn.parallel.DistributedDataParallel(model_l1, device_ids=[args.gpu])
        model_without_ddp_l1 = model_l1.module
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])
        model_without_ddp_teacher = model_teacher.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds_l0 = get_coco_api_from_dataset(dataset_val_l0)
        base_ds_l1 = get_coco_api_from_dataset(dataset_val_l1)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp_l0.detr.load_state_dict(checkpoint['model'])
        model_without_ddp_l1.detr.load_state_dict(checkpoint['model'])
        model_without_ddp_teacher.detr.load_state_dict(checkpoint['model'])


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys_l0, unexpected_keys_l0 = model_without_ddp_l0.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys_l0 = [k for k in unexpected_keys_l0 if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys_l0) > 0:
            print('Missing Keys l0: {}'.format(missing_keys_l0))
        if len(unexpected_keys_l0) > 0:
            print('Unexpected Keys l0: {}'.format(unexpected_keys_l0))

        missing_keys_l1, unexpected_keys_l1 = model_without_ddp_l1.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys_l1 = [k for k in unexpected_keys_l1 if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys_l1) > 0:
            print('Missing Keys l1: {}'.format(missing_keys_l1))
        if len(unexpected_keys_l1) > 0:
            print('Unexpected Keys l1: {}'.format(unexpected_keys_l1))

        missing_keys_teacher, unexpected_keys_teacher = model_without_ddp_teacher.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys_teacher = [k for k in unexpected_keys_teacher if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys_teacher) > 0:
            print('Missing Keys teacher: {}'.format(missing_keys_teacher))
        if len(unexpected_keys_teacher) > 0:
            print('Unexpected Keys teacher: {}'.format(unexpected_keys_teacher))

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
                                                       data_loader_val, base_ds, device, args.output_dir, args.label_map)

        return

    print("Start training")
    start_time = time.time()

    # for epoch in range(args.start_epoch, args.epochs_l0):
    #     if args.distributed:
    #         sampler_train_l0.set_epoch(epoch)
    #     train_stats = train_one_epoch0(
    #         model_l0, model_teacher, criterion_l0, data_loader_train_l0, optimizer_l0, device, epoch, args.clip_max_norm)
    #     lr_scheduler_l0.step()
    #
    #     if args.output_dir:
    #         checkpoint_paths = [output_dir / 'checkpoint.pth']
    #         # extra checkpoint before LR drop and every 5 epochs
    #         if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
    #             checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
    #         for checkpoint_path in checkpoint_paths:
    #             utils.save_on_master({
    #                 'model_l0': model_without_ddp_l0.state_dict(),
    #                 'model_l1': model_without_ddp_l1.state_dict(),
    #                 'optimizer': optimizer_l0.state_dict(),
    #                 'lr_scheduler': lr_scheduler_l0.state_dict(),
    #                 'epoch': epoch,
    #                 'args': args,
    #             }, checkpoint_path)
    #
    # missing_keys_l1, unexpected_keys_l1 = model_without_ddp_l1.load_state_dict(model_without_ddp_l0.state_dict(), strict=False)
    # unexpected_keys_l1 = [k for k in unexpected_keys_l1 if not (k.endswith('total_params') or k.endswith('total_ops'))]
    # if len(missing_keys_l1) > 0:
    #     print('Missing Keys l1: {}'.format(missing_keys_l1))
    # if len(unexpected_keys_l1) > 0:
    #     print('Unexpected Keys l1: {}'.format(unexpected_keys_l1))

    for epoch in range(args.start_epoch, args.epochs_l1):
        if args.distributed:
            sampler_train_l1.set_epoch(epoch)

        train_stats = train_one_epoch1(
            model_l1, model_l0, criterion_l1, data_loader_train_l1, optimizer_l1, device, epoch, args.clip_max_norm)
        lr_scheduler_l1.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model_l0': model_without_ddp_l0.state_dict(),
                    'model_l1': model_without_ddp_l1.state_dict(),
                    'optimizer': optimizer_l1.state_dict(),
                    'lr_scheduler': lr_scheduler_l1.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              # **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}
        #
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
