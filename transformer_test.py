# !usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:不堪沉沦
@Blog(个人博客地址): https://blog.csdn.net/qq_37344125
 
@File:transformer_test.py.py
@Time:2021/8/25 15:10
"""

import sys
sys.path.append("./")
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import time
import math
from util.utils import eval_regress, write_log, plot_results
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# 110-model ,105 best result
# 100-model ,95 best result
input_window = 95  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_inout_sequences(input_data, tw):
    """
        根据已有数据构建数据序列
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_test_data(train_data_path, test_data_path):

    # load data form test.txt
    train_series = read_csv(train_data_path, header=0, index_col=0, parse_dates=True, squeeze=True)
    test_series = read_csv(test_data_path, header=0, index_col=0, parse_dates=True, squeeze=True)

    # normalizing input values
    attr = r"Lane 1 Flow (Veh/5 Minutes)"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_series[attr].values.reshape(-1, 1))
    flow2 = scaler.transform(test_series[attr].to_numpy().reshape(-1, 1)).reshape(-1)

    test_data = create_inout_sequences(flow2, input_window)
    test_data = test_data[:-output_window]

    return test_data.to(device), scaler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


if __name__ == "__main__":
    # 1. 加载模型
    model_path = r"checkpoint/transformer-100-model.pth"
    transformer = torch.load(model_path)

    # 2. 加载测试数据
    train_data_path = r"train.txt"
    test_data_path = r"test.txt"
    test_data, scaler = get_test_data(train_data_path, test_data_path)  # 已经按照训练集数据的分布，进行归一化

    # 3.评估模型
    batch_size = 1000
    test_x, test_y = get_batch(test_data, 1, batch_size)

    output = transformer(test_x)
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    predicted = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
    test_y = torch.cat((truth, test_y[-1].view(-1).cpu()), 0)
    predicted = scaler.inverse_transform(predicted.detach().numpy().reshape(-1, 1)).reshape(1, -1)[0]
    test_y = scaler.inverse_transform(test_y.detach().numpy().reshape(-1, 1)).reshape(1, -1)[0]
    # if use scaler.inverse_transform，must uses .detach().numpy() as variable into eval_regress
    # eval_metrics = eval_regress(test_y.detach().numpy(), predicted.detach().numpy())
    eval_metrics = eval_regress(test_y, predicted)

    # 保存日志文件
    log_path = r'result/metric_log.txt'
    write_log(eval_metrics, log_path)

    # 取出288分数据，作为测试
    y_pred = predicted[:288]
    y_true = test_y[:288]
    result_pic_path = 'result/begin_img/'
    # plot_results(y_pred.detach().numpy(), y_true.detach().numpy(), result_pic_path)
    plot_results(y_pred, y_true, result_pic_path)
