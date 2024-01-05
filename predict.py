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
# from tqdm import tqdm

from nnmodel import *


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
        for result_ in result:
            writer.writerow(result_)


def predict_loader(xlsx_file: str):
    df = pd.read_excel(xlsx_file)
    x = df.to_numpy()
    # >>> read as: c, df, f, r, re1, im1, re2, im2, lambda, n_s, k0
    # >>> transpose to: df, k0, lambda, n_s, f, re1, im1, re2, im2, r, c
    return x[:, [1, -1, -3, -2, 2, 4, 5, 6, 7, 3, 0]].reshape(-1, 1, 11)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict and save results.")
    parser.add_argument("--input", type=str, default="./result.xlsx", help="Your input data file in xlsx format.")
    parser.add_argument("--output", type=str, default="./prediction_output", help="Custom save path for the results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.mkdir(args.output) if not os.path.exists(args.output) else None

    qzh = QzhResConv1D(BasicBlock,
                       [2, 2, 2, 2, 0],  # res19
                       # [3, 4, 6, 3, 0],  # res35
                       # [3, 4, 14, 3, 0],  # res51
                       # [3, 4, 23, 3, 0],  # res102
                       4).to(device)

    checkpoint = torch.load(r'.\qzh_weights\best.pth')
    qzh.load_state_dict(checkpoint["model_state_dict"])

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

        predict_x = torch.tensor(predict_loader(args.input), dtype=torch.float32)
        
        # Caution! Here are the same _mean and _var as in the nnmodel.py file.
        _mean = torch.tensor([2.0811e-03, 1.2477e-03, 9.1424e-01, 2.7408e-01, 3.8677e-04, 2.0142e-03,
                              7.9816e-04, 1.6304e-03, 1.0725e-05, 2.0788e-02, 9.2257e-04], dtype=torch.float32)
        _var = torch.tensor([1.9658e-06, 6.9116e-07, 2.0229e-02, 6.8132e-02, 3.8409e-07, 4.5327e-06,
                             7.3582e-07, 2.9000e-06, 1.1584e-09, 2.3134e-04, 5.2825e-07], dtype=torch.float32)
        
        predict_x = (predict_x - _mean) / torch.sqrt(_var)  # Standardize to normal distribution
        predict_x = predict_x.to(device)
        predictions = qzh(predict_x)
        # todo 3: figure out in what format does this model output
        # >>> (rows, 4)

    result_path = args.output
    # todo 2: Write in the input data and output data, in what format?
    write_results_to_csv(predictions.cpu().tolist(), get_next_file_number(result_path), result_path)
    print('Reasoning completion.')
