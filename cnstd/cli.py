# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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

import os
import click
import json
import time
import glob

from pprint import pformat
import cv2
import numpy as np
import torchvision.transforms as T

from .utils import rotate_page
from .consts import MODEL_VERSION, MODEL_CONFIGS, AVAILABLE_MODELS
from .utils import (
    set_logger,
    data_dir,
    load_model_params,
    imsave,
    read_img,
    pil_to_numpy,
    plot_for_debugging,
)
from .datasets import StdDataModule
from .trainer import PlTrainer, resave_model
from .model import gen_model
from . import CnStd, LayoutAnalyzer
from .yolov7.consts import CATEGORY_DICT

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}

logger = set_logger(log_level='INFO')
DEFAULT_MODEL_NAME = 'db_shufflenet_v2_small'


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option(
    '-m',
    '--model-name',
    type=click.Choice(MODEL_CONFIGS.keys()),
    default=DEFAULT_MODEL_NAME,
    help='模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-i',
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option('--train-config-fp', type=str, required=True, help='训练使用的json配置文件')
@click.option(
    '-r', '--resume-from-checkpoint', type=str, default=None, help='恢复此前中断的训练状态，继续训练'
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的模型，作为初始模型。优先级低于"--restore-training-fp"，当传入"--restore-training-fp"时，此传入失效',
)
def train(
    model_name, index_dir, train_config_fp, resume_from_checkpoint, pretrained_model_fp
):
    """训练文本检测模型"""
    logger = set_logger(log_level='DEBUG')
    train_config = json.load(open(train_config_fp))
    fpn_type = train_config.get('fpn_type', 'fpn')
    kwargs = dict(
        rotated_bbox=train_config['rotated_bbox'],
        auto_rotate_whole_image=train_config.get('auto_rotate_whole_image', False),
        pretrained_backbone=True,
        fpn_type=fpn_type,
    )
    if 'db_shufflenet_v2' in model_name:
        kwargs['pretrained_backbone'] = False
    if 'resized_shape' in train_config:
        kwargs['input_shape'] = train_config['resized_shape']
    model = gen_model(model_name, **kwargs)
    logger.info(model)
    logger.info(model.cfg)
    expected_img_shape = model.cfg['input_shape']

    train_transform = T.Compose(  # MUST NOT include `Resize`
        [
            T.RandomInvert(p=0.1),
            T.RandomPosterize(bits=4, p=0.05),
            T.RandomAdjustSharpness(sharpness_factor=0.05, p=0.1),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            T.RandomEqualize(p=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.05),
        ]
    )
    val_transform = None

    data_mod = StdDataModule(
        index_dir=index_dir,
        resized_shape=expected_img_shape[1:],
        preserve_aspect_ratio=train_config['preserve_aspect_ratio'],
        data_root_dir=train_config['data_root_dir'],
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
        debug=train_config.get('debug', False),
    )

    # train_ds = data_mod.train
    # for i in range(100):
    #     visualize_example(train_ds[i], 'debugs/train-1-%d' % i)
    #     visualize_example(train_ds[i], 'debugs/train-2-%d' % i)
    #     visualize_example(train_ds[i], 'debugs/train-3-%d' % i)
    # val_ds = data_mod.val
    # for i in range(10):
    #     visualize_example(val_ds[i], 'debugs/val-1-%d' % i)
    #     visualize_example(val_ds[i], 'debugs/val-2-%d' % i)
    #     visualize_example(val_ds[i], 'debugs/val-2-%d' % i)
    # return

    trainer = PlTrainer(
        train_config, ckpt_fn=['cnstd', 'v%s' % MODEL_VERSION, model_name, fpn_type]
    )

    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


def visualize_example(example, fp_prefix):
    image = example['image']
    imsave(image, '%s-image.jpg' % fp_prefix, normalized=True)

    def _vis_bool(img, fp):
        img *= 255
        imsave(img, fp, normalized=False)

    _vis_bool(example['gt'].transpose(1, 2, 0), '%s-gt.jpg' % fp_prefix)
    _vis_bool(np.expand_dims(example['mask'], -1), '%s-mask.jpg' % fp_prefix)
    _vis_bool(
        np.expand_dims(example['thresh_map'], -1), '%s-thresh-map.jpg' % fp_prefix
    )
    _vis_bool(
        np.expand_dims(example['thresh_mask'], -1), '%s-thresh-mask.jpg' % fp_prefix
    )


MODELS, _ = zip(*AVAILABLE_MODELS.all_models())
MODELS = sorted(MODELS)


@cli.command('predict')
@click.option(
    '-m',
    '--model-name',
    type=click.Choice(MODELS),
    default=DEFAULT_MODEL_NAME,
    help='模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-b',
    '--model-backend',
    type=click.Choice(['pytorch', 'onnx']),
    default='onnx',
    help='模型类型。默认值为 `onnx`',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option("-r", "--rotated-bbox", is_flag=True, help="是否检测带角度（非水平和垂直）的文本框")
@click.option(
    "--resized-shape",
    type=str,
    default='768,768',
    help='格式："height,width"; 预测时把图片resize到此大小再进行预测。两个值都需要是32的倍数。默认为 `768,768`',
)
@click.option(
    "--box-score-thresh", type=float, default=0.3, help="检测结果只保留分数大于此值的文本框。默认值为 `0.3`"
)
@click.option(
    "--preserve-aspect-ratio",
    type=bool,
    default=True,
    help="resize时是否保留图片原始比例。默认值为 `True`",
)
@click.option(
    "--context",
    help="使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option("-i", "--img-file-or-dir", help="输入图片的文件路径或者指定的文件夹")
@click.option(
    "-o", "--output-dir", default='./predictions', help="检测结果存放的文件夹。默认为 `./predictions`"
)
def predict(
    model_name,
    model_backend,
    pretrained_model_fp,
    rotated_bbox,
    resized_shape,
    box_score_thresh,
    preserve_aspect_ratio,
    context,
    img_file_or_dir,
    output_dir,
):
    """预测单个文件，或者指定目录下的所有图片"""
    std = CnStd(
        model_name,
        model_backend=model_backend,
        model_fp=pretrained_model_fp,
        rotated_bbox=rotated_bbox,
        context=context,
    )

    resized_shape = list(map(int, resized_shape.split(',')))  # [H, W]
    if len(resized_shape) == 1:
        resized_shape.append(resized_shape[0])

    # process image
    if os.path.isfile(img_file_or_dir):
        img_list = [img_file_or_dir]
    elif os.path.isdir(img_file_or_dir):
        fnames = glob.glob1(img_file_or_dir, "*g")
        img_list = [os.path.join(img_file_or_dir, fn) for fn in fnames]
    else:
        raise TypeError(
            'param "image_dir": %s is neither a file or a dir' % img_file_or_dir
        )

    start_time = time.time()
    std_out = std.detect(
        img_list,
        resized_shape=tuple(resized_shape),
        preserve_aspect_ratio=preserve_aspect_ratio,
        box_score_thresh=box_score_thresh,
    )
    time_cost = time.time() - start_time
    logger.info(
        '%d files are detected by cnstd, total time cost: %f, mean time cost: %f'
        % (len(img_list), time_cost, time_cost / len(img_list))
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, img_fp in enumerate(img_list):
        angle = std_out[idx]['rotated_angle']
        pil_img = read_img(img_fp)
        img = pil_to_numpy(pil_img).transpose((1, 2, 0)).astype(np.uint8)
        rotated_img = np.ascontiguousarray(rotate_page(img, -angle))

        fname = os.path.basename(img_fp).rsplit('.', maxsplit=1)[0]
        prefix_fp = os.path.join(output_dir, fname)
        plot_for_debugging(
            rotated_img, std_out[idx]['detected_texts'], box_score_thresh, prefix_fp
        )

    try:
        from cnocr import CnOcr

        ocr = CnOcr()

        logger.info('use cnocr to predict texts from the FIRST image, as an example')
        cropped_img_list = [
            box_info['cropped_img'] for box_info in std_out[0]['detected_texts']
        ]
        start_time = time.time()
        ocr_out = ocr.ocr_for_single_lines(cropped_img_list, batch_size=1)
        time_cost = time.time() - start_time
        logger.info(
            '%d cropped text boxes are recognized by cnocr, total time cost: %f, mean time cost: %f'
            % (len(cropped_img_list), time_cost, time_cost / len(cropped_img_list))
        )
        logger.info('ocr result: \n%s' % pformat(ocr_out))
    except ModuleNotFoundError as e:
        logger.warning(e)


@cli.command('resave')
@click.option('-i', '--input-model-fp', type=str, required=True, help='输入的模型文件路径')
@click.option('-o', '--output-model-fp', type=str, required=True, help='输出的模型文件路径')
def resave_model_file(
    input_model_fp, output_model_fp,
):
    """训练好的模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小"""
    resave_model(input_model_fp, output_model_fp, map_location='cpu')


@cli.command('analyze')
@click.option(
    '-m',
    '--model-name',
    type=str,
    default='mfd',
    help='模型类型。`mfd` 表示数学公式检测，`layout` 表示版面分析；默认为：`mfd`',
)
@click.option(
    '-t',
    '--model-type',
    type=str,
    default='yolov7_tiny',
    help='模型类型。当前支持 [`yolov7_tiny`, `yolov7`]',
)
@click.option(
    '-b',
    '--model-backend',
    type=click.Choice(['pytorch', 'onnx']),
    default='pytorch',
    help='模型后端架构。当前仅支持 `pytorch`',
)
@click.option(
    '-c',
    '--model-categories',
    type=str,
    default=None,
    help='模型的检测类别名称（","分割）。默认值：None，表示基于 `model_name` 自动决定',
)
@click.option(
    '-p',
    '--model-fp',
    type=str,
    default=None,
    help='使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option('-y', '--model-arch-yaml', type=str, default=None, help='模型的配置文件路径')
@click.option('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
@click.option(
    '-i', '--img-fp', type=str, default='./examples/mfd/zh.jpg', help='待分析的图片路径或图片目录'
)
@click.option(
    '-o',
    '--output-fp',
    type=str,
    default=None,
    help='分析结果输出的图片路径。默认为 `None`，会存储在当前文件夹，文件名称为输入文件名称前面增加`out-`；'
    '如输入文件名为 `img.jpg`, 输出文件名即为 `out-img.jpg`；'
    '如果输入为目录，则此路径也应该是一个目录，会将输出文件存储在此目录下',
)
@click.option(
    "--resized-shape", type=int, default=608, help='分析时把图片resize到此大小再进行。默认为 `608`',
)
@click.option(
    '--conf-thresh', type=float, default=0.25, help='Confidence Threshold。默认值为 `0.25`'
)
@click.option(
    '--iou-thresh', type=float, default=0.45, help='IOU threshold for NMS。默认值为 `0.45`'
)
def layout_analyze(
    model_name,
    model_type,
    model_backend,
    model_categories,
    model_fp,
    model_arch_yaml,
    device,
    img_fp,
    output_fp,
    resized_shape,
    conf_thresh,
    iou_thresh,
):
    """对给定图片进行 MFD 或者 版面分析。"""
    if not os.path.exists(img_fp):
        raise FileNotFoundError(img_fp)

    if model_categories is not None:
        model_categories = model_categories.split(',')
    analyzer = LayoutAnalyzer(
        model_name=model_name,
        model_type=model_type,
        model_backend=model_backend,
        model_categories=model_categories,
        model_fp=model_fp,
        model_arch_yaml=model_arch_yaml,
        device=device,
    )

    input_fp_list = []
    if os.path.isfile(img_fp):
        input_fp_list.append(img_fp)
        if output_fp is None:
            output_fp = 'out-' + os.path.basename(img_fp)
        output_fp_list = [output_fp]
    elif os.path.isdir(img_fp):
        fn_list = glob.glob1(img_fp, '*g')
        input_fp_list = [os.path.join(img_fp, fn) for fn in fn_list]
        assert (
            output_fp is not None
        ), 'output_fp should NOT be None when img_fp is a directory'
        os.makedirs(output_fp, exist_ok=True)
        output_fp_list = [os.path.join(output_fp, 'analysis-' + fn) for fn in fn_list]

    for input_fp, output_fp in zip(input_fp_list, output_fp_list):
        out = analyzer.analyze(
            input_fp,
            resized_shape=resized_shape,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
        )
        img0 = cv2.imread(input_fp, cv2.IMREAD_COLOR)
        analyzer.save_img(img0, out, output_fp)


if __name__ == '__main__':
    cli()
