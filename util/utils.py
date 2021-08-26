# !usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author:不堪沉沦
@Blog(个人博客地址): https://blog.csdn.net/qq_37344125
 
@File:utils.py
@Time:2021/8/25 20:01
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os
import time

def eval_regress(y_pred, y_true):
    """
    预测的结果评估指标的计算
    :param y_pred:
    :param y_true:
    :return:
    """
    mse = get_mse(y_pred, y_true)
    rmse = get_rmse(y_true, y_pred)
    mae = get_mae(y_true, y_pred)
    mape = get_mape(y_true, y_pred)
    smape = get_smape(y_true, y_pred)
    return mse, rmse, mae, mape, smape


def get_mse(y_true, y_pred):
    # return round(metrics.mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy()), 3)
    return round(metrics.mean_squared_error(y_true, y_pred), 3)


def get_rmse(y_true, y_pred):
    return round(np.sqrt(get_mse(y_true, y_pred)), 3)


def get_mae(y_true, y_pred):
    return round(metrics.mean_absolute_error(y_true, y_pred), 3)


def get_mape(y_true, y_pred):
    return round(np.mean(np.abs((y_pred - y_true) / y_true)) * 100, 3)


def get_smape(y_true, y_pred):
    return round(2.0 * np.mean(np.abs(y_pred - y_true) /
                               (np.abs(y_pred) + np.abs(y_true))) * 100, 3)


def plot_results(y_pred, y_true, result_pic_path):
    """Plot
    Plot the true data and predicted data.
    # Arguments
        y_pred: List/ndarray, predicted data.
        y_true: List/ndarray, ture data.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred, label="transformer")

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
    ax.set_title('epsilon=0', fontsize=12, color='r')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    if not os.path.exists(result_pic_path):
        os.mkdir(os.path.dirname(result_pic_path))

    num = open(r'./result/metric_log.txt').readlines()[-1].split(" ")[0]  # 获取运行标号
    plt.savefig(result_pic_path + "result_" + num + ".png")
    plt.show()


def write_log(eval_metrics, filepath, epsilon=0):
    with open(filepath, 'a+', encoding='utf-8') as f:
        now_time = time.strftime('%Y-%m-%d-%Hh-%Mm-%Ss', time.localtime())
        if not os.path.getsize(filepath):
            # 如果文件为空，则加入标题
            f.write(
                '%23s' % 'now_time' +
                '%20s' % 'epsilon' +
                '%10s' % 'MSE' +
                '%14s' % 'RMSE' +
                '%12s' % 'MAE' +
                '%13s' % 'MAPE' +
                '%14s' % 'SMAPE' + '\n'
            )
        lines = len(open(filepath).readlines())
        f.write(
            '%-8s' % str(lines) +
            '%10s' % now_time +
            '%11s' % str(epsilon) +
            '%14s' % np.str(eval_metrics[0]) +
            '%13s' % np.str(eval_metrics[1]) +
            '%13s' % np.str(eval_metrics[2]) +
            '%13s' % np.str(eval_metrics[3]) +
            '%13s' % np.str(eval_metrics[4]) +
            "\n"
        )


def plot_results_attack(y_pred, y_true, epsilon, result_pic_path):
    """Plot
    Plot the true data and predicted data.
    # Arguments
        y_pred: List/ndarray, predicted data.
        y_true: List/ndarray, ture data.
    """
    # attack_result_pic_path = r"../result/attack_result"  # 结果图片保存路径
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    ax.plot(x, y_pred, label="lstm")

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
    ax.set_title('epsilon='+str(epsilon), fontsize=12, color='r')


    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    if not os.path.exists(result_pic_path):
        os.mkdir(result_pic_path)
    num = open(r'./result/metric_log.txt').readlines()[-1].split(" ")[0]  # 获取运行标号
    plt.savefig(result_pic_path + "result_" + str(epsilon) + "_" + num + ".png")
    plt.show()