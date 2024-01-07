# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 10:38
# @Author  : Ahuiforever
# @File    : predict.py.py
# @Software: PyCharm

import argparse
import csv
import glob
import math
import os

import torch

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


def predict_loader(xlsx_file: str) -> np.ndarray:
    df = pd.read_excel(xlsx_file)
    x = df.to_numpy()
    x[:, [4, 5, 6, 7]] *= .01  # re, im
    x[:, [2]] *= .01  # f
    x[:, [1]] *= .1  # df
    print(f'{x.shape[0]} data are read from {xlsx_file}.')
    # >>> read as: c, df, f, r, re1, im1, re2, im2, lambda, n_s, k0
    # >>> transpose to: df, k0, lambda, n_s, f, re1, im1, re2, im2, r, c
    return x[:, [1, -1, -3, -2, 2, 4, 5, 6, 7, 3, 0]].reshape(-1, 1, 11)


def type_in_loader(typein: Union[dict, list]):
    if type(typein) is dict:
        x = [typein[key] for key in ['df', 'k0', 'lambda', 'n_s', 'f', 're1', 'im1', 're2', 'im2', 'r', 'c']]
    elif type(typein) is list:
        x = typein
    else:
        raise TypeError(f"Expect dict or list, got {type(typein)} instead.")

    return np.array(x).reshape(-1, 1, 11)


def get_last_pth():
    weight = sorted(glob.glob('./qzh_weights/*.pth'), key=os.path.getmtime, reverse=True)[0]
    print(f'Reasoning with {weight}...')
    return weight


def get_best_pth():
    weights = glob.glob('./qzh_weights/*.pth')
    weight = sorted(weights, key=lambda w: w.split('_')[2], reverse=True)[0]
    print(f'Reasoning with {weight}...')
    return weight


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict and save results.")
    parser.add_argument("--input", type=str, default="./result.xlsx",
                        help="Your input data file in xlsx format.")
    parser.add_argument("--type_in", type=str, default=None,
                        help="Type in your 11 parameters in certain format (dictionary or list). "
                             "Notice if the distribution has changed.")
    parser.add_argument("--output", type=str, default="./prediction_output",
                        help="Custom save path for the results.")
    parser.add_argument("--weight", type=str, default="./qzh_weights/best.pth",
                        help="Select the weights to use for prediction.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.mkdir(args.output) if not os.path.exists(args.output) else None

    qzh = QzhResConv1D(BasicBlock,
                       [2, 2, 2, 2, 0],  # res19
                       # [3, 4, 6, 3, 0],  # res35
                       # [3, 4, 14, 3, 0],  # res51
                       # [3, 4, 23, 3, 0],  # res102
                       4).to(device)

    checkpoint = None
    if 'best.pth' in args.weight:
        checkpoint = torch.load(get_best_pth())
    elif 'last.pth' in args.weight:
        checkpoint = torch.load(get_last_pth())
    else:
        checkpoint = torch.load(args.weight)
        print(f'Reasoning with {args.weight}...')
    qzh.load_state_dict(checkpoint["model_state_dict"])
    qzh.eval()

    # predict_data = DataReader(r'D:\Work\qzh\test')
    # predict_set = QzhData(predict_data())
    # predict_loader = DataLoader(dataset=predict_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    # todo 1. Rewrite prediction x load function.

    with torch.no_grad():
        predict_x = torch.tensor(predict_loader(args.input) if not args.type_in else type_in_loader(args.type_in),
                                 dtype=torch.float32)

        # Caution! Here are the same _mean and _var as in the nnmodel.py file.
        var_mean_sync = DataReader(r'E:\Work\qzh\all', 'result.xlsx',)
        # var_mean_sync()

        predict_x = predict_x / 3100.  # Scale to (0, 1) with L2 normalization

        predict_x = (predict_x - var_mean_sync.mean) / math.sqrt(var_mean_sync.var)
        # Standardize to normal distribution
        predict_x = predict_x.to(device)
        predictions = qzh(predict_x)
        # todo 3: figure out in what format does this model output
        # >>> (rows, 4)

    result_path = args.output
    # todo 2: Write in the input data and output data, in what format?
    write_results_to_csv(predictions.cpu().tolist(), get_next_file_number(result_path), result_path)
    print('Reasoning completion.')
