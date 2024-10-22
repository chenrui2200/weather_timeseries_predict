# -*- coding: utf-8 -*-
import pickle
from torch.utils.data import Dataset
import numpy as np
import torch

with open('./data/weather.pkl','rb') as f:
    data = pickle.load(f)
    data_train_raw, data_test_raw = [], []
    # 数据种包含 52697 条数据， 按照20 step 做一次预测，即每三个小的历史数据实现一次时序预测
    step = 20
    i = 0
    while i < 26000:
        data_train_raw.append(data[i:i + step + 1])
        i += step
    # 验证集
    i = 26000
    while i < 52000:
        data_test_raw.append(data[i:i + step + 1])
        i += step
    pass

class WeatherTrajData(Dataset):

    def __init__(self, data):
        self.data = torch.from_numpy(np.array(data)).to(torch.float32)

    def __getitem__(self, index):
        return self.data[index, :, :]

    def __len__(self):
        return self.data.size(0)
        # return data.size(0)