# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 10:38
# @Author  : Ahuiforever
# @File    : predict.py.py
# @Software: PyCharm

import argparse
import csv
import os

import pandas as pd
import torch
from tqdm import tqdm

from nnmodel import QzhLinearModel


# Function to get the next available file number in the result folder
def get_next_file_number(result_folder):
    file_list = os.listdir(result_folder)
    file_numbers = [int(file_name.split(".")[0].replace("result", ""))
                    for file_name in file_list if file_name.startswith("result")]
    if not file_numbers:
        return 1
    return max(file_numbers) + 1


# Function to write prediction results to a CSV file
def write_results_to_csv(result, file_number, result_folder):
    file_name = os.path.join(result_folder, f"result{file_number}.csv")
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)


def predict_loader(xlsx_file: str):
    df = pd.read_excel(xlsx_file)
    x = df.to_numpy()
    # >>> read as: c, df, f, r, re1, im1, re2, im2, lambda, n_s, k0
    # >>> tranpose to: df, k0, lambda, n_s, f, re1, im1, re2, im2, r, c
    # todo 4: transform x to tensor
    # * Cause in nnmodel.py the x are concatenate to the shape parameters to
    # * calculate the standard deviation and the mean, x here are not supposed
    # * to be transformed to Tensor.
    return x.transpose(1, -1, -3, -2, 2, 4, 5, 6, 7, 3, 0).reshape(-1, 11)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict and save results.")
    parser.add_argument("--input", type=str, default="./data.xlsx", help="Your input data file in xlsx format.")
    parser.add_argument("--output", type=str, default="./prediction_output", help="Custom save path for the results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qzh = QzhLinearModel().to(device)
    qzh.load_state_dict(torch.load(r'.\mlogs\best.pth'))
    qzh.eval()

    # predict_data = DataReader(r'D:\Work\qzh\test')
    # predict_set = QzhData(predict_data())
    # predict_loader = DataLoader(dataset=predict_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    # todo 1. Rewrite prediction x load function.

    # predictions = []

    with torch.no_grad():
       # for predict_idx, predict_data in tqdm(
       #         enumerate(predict_loader(args.input)),
       #         desc='Reasoning'
       # ):
       #     predict_x = predict_data
       #     prediction = qzh(predict_x)
       #    predictions.append(prediction)
            
        predict_x = torch.tensor(predict_loader(args.input),dtype=torch.float32)
        
        # Caution! Here are the same _mean and _var as in the nnmodel.py file.
        _mean = torch.tensor([2.0789e-03, 1.2479e-03, 9.1428e-01, 2.7399e-01, 3.7701e-04, 1.9827e-03,
                              7.8541e-04, 1.6054e-03, 1.0729e-05, 2.0793e-02, 9.2274e-04], dtype=torch.float32)
        _var = torch.tensor([1.9525e-06, 6.9122e-07, 2.0225e-02, 6.8118e-02, 1.1622e-07, 1.7514e-06,
                             2.7886e-07, 1.1442e-06, 1.1588e-09, 2.3138e-04, 5.2836e-07], dtype=torch.float32)
        
        predict_x = (predict_x - _mean) / torch.sqrt(_var)  # Standardize to normal distribution
        predict_x = predict_x.to(device)
        predictions = qzh(predict_x)
        # todo 3: figure out in what format does this model output
        # >>> (rows, 4)

    result_path = args.output
    # todo 2: Write in the input data and output data, in what format?
    write_results_to_csv(predictions.cpu().tolist(), get_next_file_number(result_path), result_path)
