# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# Credits: adapted from https://github.com/mindee/doctr

import torch
from torchvision.transforms import functional as F
from copy import deepcopy
import numpy as np
from typing import Tuple
from ..utils.geometry import rotate_boxes

__all__ = ["invert_colors", "rotate", "crop_detection"]


def invert_colors(img: torch.Tensor, min_val: float = 0.6) -> torch.Tensor:
    out = F.rgb_to_grayscale(img, num_output_channels=3)
    # Random RGB shift
    shift_shape = [img.shape[0], 3, 1, 1] if img.ndim == 4 else [3, 1, 1]
    rgb_shift = min_val + (1 - min_val) * torch.rand(shift_shape)
    # Inverse the color
    if out.dtype == torch.uint8:
        out = (out.to(dtype=rgb_shift.dtype) * rgb_shift).to(dtype=torch.uint8)
    else:
        out = out * rgb_shift.to(dtype=out.dtype)
    # Inverse the color
    out = 255 - out if out.dtype == torch.uint8 else 1 - out
    return out


def rotate(
    img: torch.Tensor,
    boxes: np.ndarray,
    angle: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Rotate image around the center, interpolation=NEAREST, pad with 0 (black)

    Args:
        img: image to rotate
        boxes: array of boxes to rotate as well
        angle: angle in degrees. +: counter-clockwise, -: clockwise

    Returns:
        A tuple of rotated img (tensor), rotated boxes (np array)
    """
    rotated_img = F.rotate(img, angle=angle, fill=0)  # Interpolation NEAREST by default
    _boxes = deepcopy(boxes)
    if boxes.dtype == int:
        # Compute relative boxes
        _boxes = _boxes.astype(float)
        _boxes[:, [0, 2]] = _boxes[:, [0, 2]] / img.shape[2]
        _boxes[:, [1, 3]] = _boxes[:, [1, 3]] / img.shape[1]
    # Compute rotated bboxes: xmin, ymin, xmax, ymax --> x, y, w, h, alpha
    r_boxes = rotate_boxes(_boxes, angle=angle, min_angle=0)
    if boxes.dtype == int:
        # Back to absolute boxes
        r_boxes[:, [0, 2]] *= img.shape[2]
        r_boxes[:, [1, 3]] *= img.shape[1]
    return rotated_img, r_boxes


def crop_detection(
    img: torch.Tensor,
    boxes: np.ndarray,
    crop_box: Tuple[int, int, int, int]
) -> Tuple[torch.Tensor, np.ndarray]:
    """Crop and image and associated bboxes

    Args:
        img: image to crop
        boxes: array of boxes to clip, absolute (int) or relative (float)
        crop_box: box (xmin, ymin, xmax, ymax) to crop the image. Absolute coords.

    Returns:
        A tuple of cropped image, cropped boxes, where the image is not resized.
    """
    xmin, ymin, xmax, ymax = crop_box
    croped_img = F.crop(
        img, ymin, xmin, ymax - ymin, xmax - xmin
    )
    if boxes.dtype == int:  # absolute boxes
        # Clip boxes
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], xmin, xmax)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], ymin, ymax)
    else:  # relative boxes
        h, w = img.shape[-2:]
        # Clip boxes
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], xmin / w, xmax / w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], ymin / h, ymax / h)
    # Remove 0-sized boxes
    zero_height = boxes[:, 1] == boxes[:, 3]
    zero_width = boxes[:, 0] == boxes[:, 2]
    empty_boxes = np.logical_or(zero_height, zero_width)
    boxes = boxes[~empty_boxes]

    return croped_img, boxes
