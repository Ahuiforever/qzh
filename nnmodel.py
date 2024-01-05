# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 10:16
# @Author  : Ahuiforever
# @File    : nnmodel.py
# @Software: PyCharm
"""
    This python file contains custom model, custom dataset, and aims to do logistic regression for four digits.
    ! First of all, the every data with the suffix of 'csv' is supposed to name in the following format.
    * eg: thinly_coated070_df26_195fcom079_155scom000_400nm_radius30.csv
    I name this Fully Connected Linear Logistic Regression Neural Network after Dr. Qin as QzhLinearModel. And its
structure goes as follows:

QzhLinearModel(
  (model): Sequential(
    (fc1): Linear(in_features=6, out_features=12, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=12, out_features=48, bias=True)
    (relu2): ReLU()
    (dropout1): Dropout(p=0.5, inplace=False)
    (fc3): Linear(in_features=48, out_features=48, bias=True)
    (relu3): ReLU()
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc4): Linear(in_features=48, out_features=48, bias=True)
    (relu4): ReLU()
    (dropout3): Dropout(p=0.5, inplace=False)
    (fc5): Linear(in_features=48, out_features=24, bias=True)
    (relu5): ReLU()
    (fc6): Linear(in_features=24, out_features=8, bias=True)
    (relu6): ReLU()
    (fc7): Linear(in_features=8, out_features=4, bias=True)
  )
)

"""

import os.path
import shutil
import traceback
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fnl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def predict_loader(xlsx_file: str):
    df = pd.read_excel(xlsx_file)
    x = df.to_numpy()
    # >>> read as: c, df, f, r, re1, im1, re2, im2, lambda, n_s, k0
    # >>> transpose to: df, k0, lambda, n_s, f, re1, im1, re2, im2, r, c
    # todo 4: transform x to tensor
    # * Cause in nnmodel.py the x are concatenate to the shape parameters to
    # * calculate the standard deviation and the mean, x here are not supposed
    # * to be transformed to Tensor.
    return x[:, [1, -1, -3, -2, 2, 4, 5, 6, 7, 3, 0]].reshape(-1, 11)


# ? 定义模型
class QzhLinearModel(nn.Module):
    def __init__(self):
        super(QzhLinearModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(11, 24)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(24, 96)),
            ('relu2', nn.ReLU()),
            # ('dropout1', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(96, 128)),
            ('relu3', nn.ReLU()),
            # ('dropout2', nn.Dropout(p=0.5)),
            ('fc4', nn.Linear(128, 512)),
            ('relu4', nn.ReLU()),
            # ('fc4+1', nn.Linear(128, 512)),
            # ('relu4+1', nn.ReLU()),
            # ('fc4+2', nn.Linear(512, 1024)),
            # ('relu4+2', nn.ReLU()),
            # ('fc4+3', nn.Linear(1024, 512)),
            # ('relu4+3', nn.ReLU()),
            ('fc4+4', nn.Linear(512, 128)),
            ('relu4+4', nn.ReLU()),
            # ('dropout3', nn.Dropout(p=0.5)),
            # ('fc5', nn.Linear(128, 48)),
            # ('relu5', nn.ReLU()),
            # ('fc6', nn.Linear(48, 24)),
            # ('relu6', nn.ReLU()),
            ('fc7', nn.Linear(128, 4)),
            # ('sigmoid', nn.Sigmoid()),
        ]))  # % RE-consider the rationality of the network

    def forward(self, x: any) -> any:
        x = self.model(x)
        return x


class QzhConv1D(nn.Module):
    """
    * Compute dim for 1-D convolution:
    *   out_dim = (in_dim - dilation * (kernel - 1) + 2 * padding) / stride + 1
    ? Example:
    ?   out_dim = (11 - 3 + 2*1 ) / 1 + 1 = 11
    """
    def __init__(self,):
        super().__init__()
        # todo 1. add dilation conv1d
        # todo 2. add bigger conv1d
        # todo 3. add residual connection
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, 36, 3, 1, 1)),
            ('relu1', nn.ReLU(inplace=False)),
            ('conv2', nn.Conv1d(36, 128, 3, 1, 1)),
            ('relu2', nn.ReLU(inplace=False)),
            ('conv3', nn.Conv1d(128, 512, 3, 1, 1)),
            ('relu3', nn.ReLU(inplace=False)),
            ('conv4', nn.Conv1d(512, 1024, 3, 1, 1)),
            ('relu4', nn.ReLU(inplace=False)),
            ('conv5', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5', nn.ReLU(inplace=False)),
            ('conv5+1', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+1', nn.ReLU(inplace=False)),
            ('conv5+2', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+2', nn.ReLU(inplace=False)),
            ('conv5+3', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+3', nn.ReLU(inplace=False)),
            ('conv5+4', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+4', nn.ReLU(inplace=False)),
            ('conv5+5', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+5', nn.ReLU(inplace=False)),
            ('conv5+6', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+6', nn.ReLU(inplace=False)),
            ('conv5+7', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+7', nn.ReLU(inplace=False)),
            ('conv5+8', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+8', nn.ReLU(inplace=False)),
            ('conv5+9', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+9', nn.ReLU(inplace=False)),
            ('conv5+10', nn.Conv1d(1024, 1024, 3, 1, 1)),
            ('relu5+10', nn.ReLU(inplace=False)),
            ('conv6', nn.Conv1d(1024, 512, 3, 1, 1)),
            ('relu6', nn.ReLU(inplace=False)),
            ('conv7', nn.Conv1d(512, 128, 3, 1, 1)),
            ('relu7', nn.ReLU(inplace=False)),
            ('conv8', nn.Conv1d(128, 1, 3, 1, 1)),
            ('relu8', nn.ReLU(inplace=False)),
            ('fc', nn.Linear(11, 4)),
        ]))

    def forward(self, x):
        x = self.model(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    # Expansion indicates how many times the number of channels in the last layer
    # is multiple of the that of the first layer.

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class QzhResConv1D(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock]],
                 layers: List[int],
                 num_predictions: int = 4,
                 ) -> None:
        super().__init__()
        self.inplanes = 24
        self.lastplanes = 11
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.layer5 = self._make_layer(block, self.lastplanes, layers[4])
        self.fc = nn.Linear(self.lastplanes, out_features=num_predictions)

    def _make_layer(self,
                    block: Type[Union[BasicBlock]],
                    planes: int,
                    blocks: int,
                    ) -> nn.Sequential:

        downsample = nn.Sequential(
            nn.Conv1d(self.inplanes, planes, 1, 1, 0),
            nn.BatchNorm1d(planes),
        )

        layers = [block(self.inplanes, planes, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.inplanes, planes)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


# ? 读取数据
class DataReader:
    def __init__(self, root_dir: str, contain_xlsx_file: str = ''):
        self.root_dir = root_dir
        self.file_paths = []
        self.contain_xlsx_file = contain_xlsx_file
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                self.file_path = os.path.join(root, file)
                if os.path.isfile(self.file_path) and (self.file_path[-4:] == 'xlsx' or self.file_path[-3:] == 'csv'):
                    self.file_paths.append(self.file_path)
        print(f'{len(self.file_paths)} files are read from {root_dir}.')

    def _complex(self, keyword: str) -> tuple:
        indices = self.file_path.find(keyword)
        if indices:
            re = float(self.file_path[indices - 3:indices]) * .01
            im = float(self.file_path[indices + 4:indices + 7]) * .01
        else:
            raise NameError(f"@param：Re%3-{keyword}-Im%3 not found in path {self.file_path}.")
        return re, im

    def _lambda(self) -> float:
        indices = self.file_path.find('nm')
        if indices:
            lambd = float(self.file_path[indices - 3:indices])
        else:
            raise NameError(f"@param：lambda%3-nm not found in path {self.file_path}.")
        return lambd

    def _f(self) -> float:
        indices = self.file_path.find('coated')
        if indices:
            f = int(self.file_path[indices + 6:indices + 9]) * .01
            # >>> coated075 -> f=0.75
            # self.f = float(self.file_path[self.indices_f + 6:self.indices_f + 8]) * .01  # % temp
        else:
            raise NameError(f"@param：coated-f%3 not found in path {self.file_path}.")
        return f

    def _df(self) -> float:
        indices = self.file_path.find('df')
        if indices:
            df = float(self.file_path[indices + 2:indices + 4]) * .1
        else:
            raise NameError(f"@param：df-df%2 not found in path {self.file_path}.")
        return df

    def _r(self) -> float:
        indices = self.file_path.find('radius')
        if indices:
            r = float(self.file_path[indices + 6:indices + 8])
        else:
            # self.r = 0.  # % temp
            raise NameError(f"@param：radius-r%2 not found in path {self.file_path}.")
        return r

    def _shape_parameter(self) -> np.ndarray:
        shape_parameter = np.column_stack([
            np.full_like(self.n_s_list, self.df, dtype=np.float32),
            np.full_like(self.n_s_list, self.k_0, dtype=np.float32),
            # np.full_like(self.n_s_list, self.lambd),
            self.lambd,
            self.n_s_list,
            np.full_like(self.n_s_list, self.f, dtype=np.float32),
            np.full_like(self.n_s_list, self.re1, dtype=np.float32),
            np.full_like(self.n_s_list, self.im1, dtype=np.float32),
            np.full_like(self.n_s_list, self.re2, dtype=np.float32),
            np.full_like(self.n_s_list, self.im2, dtype=np.float32),
            np.full_like(self.n_s_list, self.r, dtype=np.float32),
            np.full_like(self.n_s_list, self.c, dtype=np.float32)
        ])
        return shape_parameter

    def __call__(self) -> tuple:
        self.shape_parameters, self.labels = [], []
        for self.file_path in self.file_paths:
            # * Re1, Im1, Re2, Im2
            self.re1, self.im1 = self._complex('fcom')
            self.re2, self.im2 = self._complex('scom')

            # * C
            self.c = 1 if 'thinly' in self.file_path else 0

            # * lambda
            # self.lambd = self._lambda()
            self.data_sheet = pd.read_excel(self.file_path, header=None)
            self.lambd = self.data_sheet.iloc[:, 1]

            # * f
            self.f = self._f()

            # * Df
            self.df = self._df()

            # * k0
            self.k_0 = 1.2

            # * ns
            self.n_s_list = self.data_sheet.iloc[:, 0]  # >>> (rows, )

            # * R
            self.r = self._r()

            # * Collect all the parameters.
            shape_parameter = self._shape_parameter()
            self.shape_parameters.append(shape_parameter)
            # shape_parameter = np.array([
            #     [self.d_f, self.k_0, self.lambd, n_s, self.f, self.re1, self.im1, self.re2, self.im2, self.r, self.c]
            #     for n_s in self.n_s_list
            # ]) -> (rows, 12), rows = number of n_s

            # * Collect labels.
            label = self.data_sheet.iloc[:, 2:]
            self.labels.append(label)

        if self.contain_xlsx_file != '':
            self.shape_parameters.append(predict_loader(self.contain_xlsx_file))
        self.shape_parameters = np.concatenate(self.shape_parameters, axis=0)  # >>> (rows * for, 11)
        self.labels = np.concatenate(self.labels, axis=0)  # >>> (rows * for, 4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x = torch.tensor(self.shape_parameters, dtype=torch.float32)
        self.x = fnl.normalize(self.x, p=2, dim=1)  # Scale to (0, 1) with L2 normalization
        # __Calculate the mean and standard deviation
        # _mean = torch.mean(self.x, dim=0)
        # _var = torch.var(self.x, dim=0)
        # print(f"mean: {_mean}"
        #       f"var: {_var}")
        # __=========================================
        _mean = torch.tensor([2.0811e-03, 1.2477e-03, 9.1424e-01, 2.7408e-01, 3.8677e-04, 2.0142e-03,
                              7.9816e-04, 1.6304e-03, 1.0725e-05, 2.0788e-02, 9.2257e-04], dtype=torch.float32)
        _var = torch.tensor([1.9658e-06, 6.9116e-07, 2.0229e-02, 6.8132e-02, 3.8409e-07, 4.5327e-06,
                             7.3582e-07, 2.9000e-06, 1.1584e-09, 2.3134e-04, 5.2825e-07], dtype=torch.float32)
        self.x = (self.x - _mean) / torch.sqrt(_var)  # Standardize to normal distribution
        self.x = self.x.to(device)

        self.y = torch.tensor(self.labels, dtype=torch.float32).to(device)

        return self.x, self.y


# ? 自定义数据集
class QzhData(Dataset):
    def __init__(self, data: tuple):
        self.data = data[0]
        self.target = data[1]

    def __getitem__(self, item: int):
        x = self.data[item].reshape(1, 11)
        y = self.target[item]
        return x, y

    def __len__(self):
        return len(self.data)


# ? 路径检查器
class PathChecker:
    def __init__(self, path):
        self.path = path

    def __call__(self, del_: bool = True, **kwargs):
        try:
            self.path = kwargs['path']
        except KeyError:
            pass
        if del_:
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)


if __name__ == '__main__':

    # ! Run the main function to compute the mean and standard deviation of train set.

    train_data = DataReader(r'D:\Work\qzh\train')
    # dev_data = DataReader(r'D:\Work\qzh\dev')
    # test_data = DataReader(r'D:\Work\qzh\test')

    train_set = QzhData(train_data())
    # dev_set = QzhData(dev_data())
    # test_set = QzhData(test_data())

    # ? 加载数据
    # // todo: 1. train_set, dev_set, test_set
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=6, drop_last=False)
    # dev_loader = DataLoader(dataset=dev_set, batch_size=128, shuffle=True, num_workers=6, drop_last=False)
    # test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True, num_workers=6, drop_last=False)

    # Create a StandardScaler instance
    scaler_ = StandardScaler()

    mean = 0.0
    std = 0.0
    total_samples = 0
    for input_x, output_y in train_loader:
        batch_size = input_x.size(0)

        # # Normalize x to the scale of (0, 1)
        # normalized_x = f.normalize(input_x, p=2, dim=1)

        # Fit the scaler on the data to compute mean and standard deviation
        scaler_.fit(input_x)

        # Get the computed mean and standard deviation
        mean += scaler_.mean_
        std += scaler_.scale_

        total_samples += 1

    mean /= total_samples
    std /= total_samples

    print("Mean values:", mean)
    print("Standard deviation values:", std)
