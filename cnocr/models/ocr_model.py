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
import logging
from collections import OrderedDict
from typing import Tuple, Dict, Any, Optional, List, Union
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .ctc import CTCPostProcessor
from ..consts import ENCODER_CONFIGS, DECODER_CONFIGS
from ..data_utils.utils import encode_sequences
from .densenet import DenseNet, DenseNetLite
from .mobilenet import gen_mobilenet_v3

logger = logging.getLogger(__name__)


class EncoderManager(object):
    @classmethod
    def gen_encoder(
        cls, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Tuple[nn.Module, int]:
        if name is not None:
            assert name in ENCODER_CONFIGS
            config = deepcopy(ENCODER_CONFIGS[name])
        else:
            assert config is not None and 'name' in config
            name = config.pop('name')

        if name.lower().startswith('densenet_lite'):
            out_length = config.pop('out_length')
            encoder = DenseNetLite(**config)
        elif name.lower().startswith('densenet'):
            out_length = config.pop('out_length')
            encoder = DenseNet(**config)
        elif name.lower().startswith('mobilenet'):
            arch = config['arch']
            out_length = config.pop('out_length')
            encoder = gen_mobilenet_v3(arch)
        else:
            raise ValueError('not supported encoder name: %s' % name)
        return encoder, out_length


class DecoderManager(object):
    @classmethod
    def gen_decoder(
        cls,
        input_size: int,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, int]:
        if name is not None:
            assert name in DECODER_CONFIGS
            config = deepcopy(DECODER_CONFIGS[name])
        else:
            assert config is not None and 'name' in config
            name = config.pop('name')

        if 'lstm' in name.lower():
            decoder = nn.LSTM(
                input_size=input_size,
                hidden_size=config['rnn_units'],
                batch_first=True,
                num_layers=2,
                bidirectional=True,
            )
            out_length = config['rnn_units'] * 2
        elif 'gru' in name.lower():
            decoder = nn.GRU(
                input_size=input_size,
                hidden_size=config['rnn_units'],
                batch_first=True,
                num_layers=config.get('num_layers', 2),
                bidirectional=True,
            )
            out_length = config['rnn_units'] * 2
        elif 'fc' in name.lower():
            decoder = nn.Sequential(
                nn.Dropout(p=config['dropout']),
                # nn.Tanh(),
                nn.Linear(input_size, config['hidden_size']),
                nn.Dropout(p=config['dropout']),
                nn.Tanh(),
            )
            out_length = config['hidden_size']
        else:
            raise ValueError('not supported decoder name: %s' % name)
        return decoder, out_length


class OcrModel(nn.Module):
    """OCR Model.

    Args:
        encoder: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        cfg: configuration dictionary
    """

    _children_names: List[str] = [
        'encoder',
        'decoder',
        'linear',
        'postprocessor',
    ]

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        decoder_out_length: int,
        vocab: List[str],
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.letter2id = {letter: idx for idx, letter in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.letter2id)

        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder_out_length, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    @classmethod
    def from_name(cls, name: str, vocab: List[str]):
        encoder_name, decoder_name = name.split('-')[-2:]
        encoder, encoder_out_len = EncoderManager.gen_encoder(encoder_name)
        decoder, decoder_out_len = DecoderManager.gen_decoder(
            encoder_out_len, decoder_name
        )
        return cls(encoder, decoder, decoder_out_len, vocab)

    def calculate_loss(
        self, batch, return_model_output: bool = False, return_preds: bool = False,
    ):
        imgs, img_lengths, labels_list, label_lengths = batch
        # print(f'cal loss before: {labels_list=}, {img_lengths=}, {labels_list=}, {label_lengths=}')
        return self(
            imgs, img_lengths, labels_list, None, return_model_output, return_preds
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        target: Optional[List[str]] = None,
        candidates: Optional[Union[str, List[str]]] = None,
        return_logits: bool = True,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """

        :param x: [B, 1, H, W]; 一组padding后的图片
        :param input_lengths: shape: [B]；每张图片padding前的真实长度（宽度）
        :param target: 真实的字符串
        :param candidates: None or candidate strs; 允许的候选字符集合
        :param return_logits: 是否返回预测的logits值
        :param return_preds: 是否返回预测的字符串
        :return: 预测结果
        """
        # print(f'forward before: {x.shape=}, {x.min()=}, {x.max()=}')
        features = self.encoder(x)
        # print(f'forward encoder: {features.shape=}, {features.min()=}, {features.max()=}, {input_lengths=}')
        input_lengths = torch.div(
            input_lengths, self.encoder.compress_ratio, rounding_mode='floor'
        )
        # print(f'forward div: {input_lengths=}')
        if torch.any(input_lengths < 1).item():
        # if input_lengths.min() < 1:
            logger.error(f'input_lengths min: {input_lengths.min()}, {input_lengths=}')
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)  # B x W x C*H
        # print(f'forward reshape: {features_seq.shape=}, {features_seq.min()=}, {features_seq.max()=}')

        logits = self._decode(features_seq, input_lengths)
        # print(f'forward decode: {logits.shape=}, {logits.min()=}, {logits.max()=}')

        logits = self.linear(logits)
        logits = self.mask_by_candidates(logits, candidates, self.vocab, self.letter2id)
        # print(f'forward linear: {logits.shape=}, {logits.min()=}, {logits.max()=}')

        out: OrderedDict[str, Any] = {}
        if return_logits:
            out["logits"] = logits
        out['output_lengths'] = input_lengths

        if target is None or return_preds:
            # Post-process boxes
            if self.postprocessor is not None:
                out["preds"] = self.postprocessor(logits, input_lengths)
                # print(f'forward postprocessor: {out["preds"]=}')

        if target is not None:
            out['loss'] = self._compute_loss(logits, target, input_lengths)
            # print(f'forward loss: {out["loss"]=}')

        out['target'] = target
        return dict(out)

    def _decode(self, features_seq, input_lengths):
        if not isinstance(self.decoder, (nn.LSTM, nn.GRU)):
            return self.decoder(features_seq)

        w = features_seq.shape[1]
        features_seq = pack_padded_sequence(
            features_seq,
            input_lengths.to(device='cpu'),
            batch_first=True,
            enforce_sorted=False,
        )
        logits, _ = self.decoder(features_seq)
        logits, output_lens = pad_packed_sequence(
            logits, batch_first=True, total_length=w
        )
        return logits

    @classmethod
    def mask_by_candidates(
        cls,
        logits: torch.Tensor,
        candidates: Optional[Union[str, List[str]]],
        vocab: List[str],
        letter2id: Dict[str, int],
    ):
        if candidates is None:
            return logits

        _candidates = [letter2id[word] for word in candidates]
        _candidates.sort()
        _candidates = torch.tensor(_candidates, dtype=torch.int64)

        candidates = torch.zeros(
            (len(vocab) + 1,), dtype=torch.bool, device=logits.device
        )
        candidates[_candidates] = True
        candidates[-1] = True  # 间隔符号/填充符号，必须为真
        candidates = candidates.unsqueeze(0).unsqueeze(0)  # 1 x 1 x (vocab_size+1)
        logits.masked_fill_(~candidates, -100.0)
        return logits

    def _compute_loss(
        self,
        model_output: torch.Tensor,
        target: List[str],
        seq_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_length: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.compute_target(target)

        if seq_length is None:
            batch_len = model_output.shape[0]
            seq_length = model_output.shape[1] * torch.ones(
                size=(batch_len,), dtype=torch.int32
            )

        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)

        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt).to(device=probs.device),
            seq_length,
            torch.tensor(seq_len, dtype=torch.int, device=probs.device),
            len(self.vocab),
            zero_infinity=True,
        )

        return ctc_loss

    def compute_target(self, gts: List[str],) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(
            sequences=gts, vocab=self.letter2id, eos=len(self.letter2id),
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len
