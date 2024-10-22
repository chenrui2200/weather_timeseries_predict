# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_read import data_train_raw, WeatherTrajData
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

log_writer = SummaryWriter()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('torch.cuda.is_available=' + str(torch.cuda.is_available()))
torch.set_default_tensor_type(torch.FloatTensor)


# torch.nn.Transformer

def PredictTrajectory(tra_pred, tra_true):

    idx = random.randrange(0, tra_true.shape[0])
    pred = tra_pred[idx, :, :].cpu().detach().numpy()
    true = tra_true[idx, :, :].cpu().detach().numpy()

    print('------------------------------')
    print('real value  rho {0}, ' +
          'sh {1}, ' +
          'T {2}， ' +
          'Tdew {3}， ' +
          'Tlog {4}， ' +
          'Tpot {5}， ' +
          'VPact {6}， ' +
          'VPmax {7}'
          .format(true[:, 0][true[:, 0].shape[0] - 1],
                  true[:, 1][true[:, 1].shape[0] - 1],
                  true[:, 2][true[:, 2].shape[0] - 1],
                  true[:, 3][true[:, 3].shape[0] - 1],
                  true[:, 4][true[:, 4].shape[0] - 1],
                  true[:, 5][true[:, 5].shape[0] - 1],
                  true[:, 6][true[:, 6].shape[0] - 1],
                  true[:, 7][true[:, 7].shape[0] - 1],
                  ))
    print('\n')
    # 按照data_read里顺序，下一时刻的位置预测为
    print('predict value，rho {0}, ' +
          'sh {1}, ' +
          'T {2}， ' +
          'Tdew {3}， ' +
          'Tlog {4}， ' +
          'Tpot {5}， ' +
          'VPact {6}， ' +
          'VPmax {7}'
          .format(pred[:, 0][pred[:, 0].shape[0] - 1],
                  pred[:, 1][pred[:, 1].shape[0] - 1],
                  pred[:, 2][pred[:, 2].shape[0] - 1],
                  pred[:, 3][pred[:, 3].shape[0] - 1],
                  pred[:, 4][pred[:, 4].shape[0] - 1],
                  pred[:, 5][pred[:, 5].shape[0] - 1],
                  pred[:, 6][pred[:, 6].shape[0] - 1],
                  pred[:, 7][pred[:, 7].shape[0] - 1],
                  ))


def cal_performance(tra_pred, tra_true):
    return F.mse_loss(tra_pred, tra_true)


def train(model, dataloader, optimizer, device, opt):
    for id, epoch_i in enumerate(tqdm.tqdm(range(opt.epoch))):
        model.train()
        total_loss = 0
        for idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            tra_pred = model(input_data=data.to(device).to(torch.float32), device=device)
            # backward and update parameters
            loss = cal_performance(tra_pred, data[:, 1:, :].to(device).to(torch.float32))
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()

        log_writer.add_scalar("loss", total_loss, epoch_i)
        log_writer.add_scalar("lr", optimizer.get_lr(), epoch_i)
        if epoch_i % 100 == 0:
            print("epoch = %d, epoch_loss= %lf ,total_loss = %lf" % (epoch_i, loss.item(), total_loss))

    torch.save(model, 'model.pt')
    # another method to save model
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.get_state_dict(),
        "epoch": epoch_i
    }

    if not os.path.isdir("./checkpoint"):
        os.mkdir("./checkpoint")
    torch.save(checkpoint, "./checkpoint/ckpt.pth")

    print("Train Finish")


def test(model, dataloader, device):
    total_loss = 0
    for idx, data in enumerate(dataloader):
        tra_pred = model(input_data=data.to(device).to(torch.float32), device=device)
        loss = cal_performance(tra_pred, data[:, 1:, :].to(device).to(torch.float32))
    total_loss += loss.item()
    PredictTrajectory(tra_pred, data[:, 0:20, :])
    print("Test Finish, total_loss = {}".format(total_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=10000)
    parser.add_argument('-b', '--batch_size', type=int, default=1300)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_head', type=int, default=2)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-do_train', type=bool, default=False)
    parser.add_argument('-do_retrain', type=bool, default=False)
    parser.add_argument('-do_eval', type=bool, default=True)

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    if torch.cuda.is_available():
        # 使用 CUDA 设备
        device = torch.device("cuda")
    else:
        # 使用 CPU 设备
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model_train = Transformer(
        500,
        500,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
    ).to(device)

    if opt.do_train == True:
        # data=torch.from_numpy(np.array(data_train)).to(device).to(torch.float32)
        data_train = WeatherTrajData(data_train_raw)
        train_loader = DataLoader(dataset=data_train, batch_size=opt.batch_size, shuffle=False)
        parameters = model_train.parameters()
        optimizer = ScheduledOptim(
            optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-09),
            opt.lr, opt.d_model, opt.n_warmup_steps)

        if opt.do_retrain == True:
            checkpoint = torch.load("./checkpoint/ckpt.pth")
            model_train.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        start_time = time.time()
        train(
            model=model_train,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            opt=opt
        )
        end_time = time.time()
        print("train time = {} seconds".format(end_time - start_time))

    if opt.do_eval == True:
        data_test = WeatherTrajData(data_train_raw)
        test_loader = DataLoader(dataset=data_test, batch_size=opt.batch_size, shuffle=False)
        #model = torch.load('model.pt').to(device)
        model = torch.load('model.pt', map_location=torch.device('cpu'))

        test(
            model=model,
            dataloader=test_loader,
            device=device
        )
