# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .deformable_detr_base import build as build_base
from .deformable_detr_data import build as build_data
from .deformable_detr_local0 import build as build_local0
from .deformable_detr_local1 import build as build_local1
from .deformable_detr_teacher import build as build_teacher

def build_model(args):
    return build_base(args)
def build_model_data(args):
    return build_data(args)
def build_model0(args):
    return build_local0(args)
def build_model1(args):
    return build_local1(args)
def build_model_teacher(args):
    return build_teacher(args)
