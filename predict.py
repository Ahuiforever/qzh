# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 10:38
# @Author  : Ahuiforever
# @File    : predict.py.py
# @Software: PyCharm

import argparse
import csv
import os

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict and save results.")
    parser.add_argument("--input", type=str, default="./data.csv", help="Your input data file.")
    parser.add_argument("--result", type=str, default="./prediction results", help="Custom save path for the results")
    args = parser.parse_args()

    qzh = QzhLinearModel()
    qzh.load_state_dict(torch.load(r'.\mlogs\best.pth'))
    qzh.eval()

    # predict_data = DataReader(r'D:\Work\qzh\test')
    # predict_set = QzhData(predict_data())
    # predict_loader = DataLoader(dataset=predict_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    # todo 1. Rewrite prediction x load function.

    predictions = []

    with torch.no_grad():
        for predict_idx, predict_data in tqdm(
                enumerate(predict_loader),
                desc='Reasoning'
        ):
            predict_x, predict_target_y = predict_data
            prediction = qzh(predict_x)
            predictions.append(prediction)

    result_path = args.result
    # todo 2: Write in the input data and output data, in what format?
    write_results_to_csv(predictions, get_next_file_number(result_path), result_path)
