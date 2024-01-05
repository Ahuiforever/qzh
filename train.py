# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 14:30
# @Author  : Ahuiforever
# @File    : train.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from nnmodel import *
from utils import ModelSaver


def save_curve(input_cpu_data1: np.ndarray,
               input_cpu_data2: np.ndarray,
               _epoch: int,
               train: bool,
               ) -> None:
    labels = ['Qext', 'Qabs', 'Qsca', 'ASY']
    colors = ['red', 'green', 'blue', 'cyan']
    fig = plt.figure(figsize=(20, 15))
    axs = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]
    fig.suptitle('Four Parameters: Prediction--curve, GroundTruth--dots')

    for idx, label in enumerate(labels):
        seed = np.random.randint(0, 100, size=1)[0]
        np.random.seed(seed)
        try:
            x = np.sort(
                np.random.choice(
                    np.arange(input_cpu_data1.shape[0]),
                    size=100,
                    replace=False),
                kind='quicksort'
            )
        except ValueError:
            x = np.sort(
                np.random.choice(
                    np.arange(input_cpu_data1.shape[0]),
                    size=input_cpu_data1.shape[0]-1,
                    replace=False),
                kind='quicksort')
        curves = input_cpu_data1[:, idx]
        dots = input_cpu_data2[:, idx]
        axs[idx].plot('x', 'y', '',
                      data={'x': np.arange(x.shape[0]),
                            'y': curves[x]},
                      color=colors[idx], linewidth=0.5)
        axs[idx].scatter(x=np.arange(x.shape[0]),
                         y=dots[x],
                         color='black', s=2, linewidths=None)

        axs[idx].set_title(label)
        axs[idx].set_xlabel('i-th Shape Parameters')
        axs[idx].set_ylabel('Corresponding Value')
    if _epoch == 0 and train is True:
        try:
            shutil.rmtree('./plots')
        except FileNotFoundError:
            tqdm.write('Created directory ./plots.')
        os.mkdir('./plots')
    fig.savefig(f"plots/{_epoch}th_{'train' if train else 'Val'}_Fit.png")
    plt.close()


if __name__ == '__main__':
    # ? 实例化路径检查器
    log_path = './tensorboard'
    pc = PathChecker(log_path)
    pc(del_=True)

    # ? 日志板
    writer = SummaryWriter(log_path)
    # $ tensorboard --logdir="qzh/tensorboard" --port=6007

    # ? 实例化模型
    # qzh = QzhLinearModel()
    # qzh = QzhConv1D()
    qzh = QzhResConv1D(BasicBlock,
                       [2, 2, 2, 2, 0],  # res19
                       # [3, 4, 6, 3, 0],  # res35
                       # [3, 4, 14, 3, 0],  # res51
                       # [3, 4, 23, 3, 0],  # res102
                       4)

    # ? 实例化数据读取器, 实例化数据集
    train_data = DataReader(r'E:\Work\qzh\train',
                            # 'result.xlsx',
                            )
    dev_data = DataReader(r'E:\Work\qzh\dev')
    # test_data = DataReader(r'D:\Work\qzh\test')

    train_set = QzhData(train_data())
    dev_set = QzhData(dev_data())
    # test_set = QzhData(test_data())

    # ? 加载数据
    # // todo: 1. train_set, dev_set, test_set
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
    dev_loader = DataLoader(dataset=dev_set, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
    # test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True, num_workers=6, drop_last=False)

    # ? 定义回归损失函数 | L1Loss, SmoothL1Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_function = nn.L1Loss().to(device)
    loss_function = nn.MSELoss().to(device)
    # loss_function = nn.CrossEntropyLoss().to(device)
    # loss_function = nn.SmoothL1Loss()
    # * If the task is a multi-digit prediction task using logistic regression,
    # * where the goal is to predict multiple digits, you can consider using the Mean Squared Error (MSE) loss function.
    # * The MSE loss measures the average squared difference between the predicted values and the true values.

    # ? 定义优化器 | Adagrad, Adam, RMSprop, SGD
    optimizer = optim.AdamW(qzh.parameters(), lr=0.001, weight_decay=1e-2)

    # ? 定义学习率衰减 | StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # ? 定义准确率的误差限 | 这里定义为 >>> abs(预测值 - 实际值) / 实际值 < tolerance 为预测正确
    tolerance = 0.2  # ! Crucial: This would affect the way to measure the accuracy of network.
    # ! But tuning this won't jeopardise the performance of it.

    # ? 实例化模型保存器
    ms = ModelSaver(
        model=qzh,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_interval=10,
        max_checkpoints_to_keep=10,
        checkpoint_dir="./qzh_weights",  # % qzh model save directory
    )

    # ? 调用GPU并训练
    qzh = qzh.to(device)
    epochs = 100  # * Number of epochs, it could be larger.
    # for epoch in range(epochs):
    for epoch in trange(epochs,
                        desc='Epochs',
                        leave=False,
                        position=0,
                        ncols=100,
                        colour='blue'):
        qzh.train()
        total_train_accuracy = 0
        total_loss = 0
        output_data = torch.tensor([]).to(device)
        target_data = torch.tensor([]).to(device)
        batch_pbar = tqdm(
            train_loader,
            desc='Batches',
            total=len(train_loader),
            leave=False,
            position=1,
            ncols=100,
            colour='green',
        )
        for train_idx, train_data in enumerate(batch_pbar):
            # for train_idx, train_data in enumerate(train_loader):
            train_x, target_y = train_data
            train_y = qzh(train_x)
            loss = loss_function(train_y, target_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct_predictions = abs(train_y - target_y) / target_y <= tolerance
            total_train_accuracy += correct_predictions.sum().item() / target_y.numel()  # numel: number of elements
            total_loss += loss

            # cache the output and target data
            output_data = torch.concatenate([output_data, train_y], dim=0)
            target_data = torch.concatenate([target_data, target_y], dim=0)

        # Calculate the Mean Absolute Percentile Error, MAPE
        output_datas = output_data.view(-1,)
        target_datas = target_data.view(-1,)
        assert output_datas.shape == target_datas.shape
        absolute_difference = abs(output_datas - target_datas)
        absolute_difference_percentile = torch.divide(absolute_difference, target_datas + 1e-5)
        train_mape = absolute_difference_percentile.mean().item()
        # Save the training prediction data and ground truth data fit curve.
        save_curve(output_data.detach().cpu().numpy(),
                   target_data.detach().cpu().numpy(),
                   epoch,
                   train=True,
                   )

        train_accuracy = total_train_accuracy / len(train_loader)
        train_loss = total_loss / len(train_loader)
        # tqdm.write(f'\033[1;35m train_accuracy {train_accuracy * 100}%\033[0m')

        # ? Development set for cross validation
        qzh.eval()
        with torch.no_grad():
            total_dev_loss = 0
            total_dev_accuracy = 0
            val_output_data = torch.tensor([]).to(device)
            val_target_data = torch.tensor([]).to(device)
            for dev_idx, dev_data in enumerate(dev_loader):
                dev_x, dev_target_y = dev_data
                dev_y = qzh(dev_x)
                dev_loss = loss_function(dev_y, dev_target_y)
                total_dev_loss += dev_loss
                dev_correct_predictions = abs(dev_y - dev_target_y) / dev_target_y <= tolerance
                total_dev_accuracy += dev_correct_predictions.sum().item() / dev_target_y.numel()
                # cache the output and target data
                val_output_data = torch.concatenate([val_output_data, dev_y], dim=0)
                val_target_data = torch.concatenate([val_target_data, dev_target_y], dim=0)

        dev_accuracy = total_dev_accuracy / len(dev_loader)
        val_loss = total_dev_loss / len(dev_loader)
        # Calculate the Mean Absolute Percentile Error, MAPE
        val_output_datas = val_output_data.view(-1,)
        val_target_datas = val_target_data.view(-1,)
        assert val_output_datas.shape == val_target_datas.shape
        val_absolute_difference = abs(val_output_datas - val_target_datas)
        val_absolute_difference_percentile = torch.divide(val_absolute_difference, val_target_datas + 1e-5)
        val_mape = val_absolute_difference_percentile.mean().item()
        # Save the Validating prediction data and ground truth data fit curve.
        save_curve(val_output_data.detach().cpu().numpy(),
                   val_target_data.detach().cpu().numpy(),
                   epoch,
                   train=False,
                   )

        # ? 写入Tensorboard
        if epoch % 30 == 0:
            tqdm.write(f'\033[1;34m val_loss {total_dev_loss}\033[0m')
            tqdm.write(f'\033[1;36m val_accuracy {dev_accuracy * 100}%\033[0m')
        # writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('accuracy', {'train': train_accuracy*100, 'val': dev_accuracy*100}, epoch)
        writer.add_scalars('Mean Absolute Percentile Error', {'train': train_mape, 'val': val_mape}, epoch)

        # ? 保存模型
        # ======================================================================= OLD VERSION
        # if epoch % 1000 == 0:
        #     pc(path='./mlogs', del_=True)
        #     if loss:
        #         torch.save(qzh.state_dict(), f'./mlogs/qzh_{epoch + 1}_{loss}_{total_dev_loss}.pth')
        #     else:
        #         torch.save(qzh.state_dict(), f'./mlogs/qzh_{epoch + 1}_undefined_{total_dev_loss}.pth')
        ms(epoch=epoch, val_loss=val_loss, val_accuracy=dev_accuracy)

        scheduler.step()
        torch.cuda.empty_cache()

    writer.close()
    print('Training is done. Looking up the tensorboard for more details.')
