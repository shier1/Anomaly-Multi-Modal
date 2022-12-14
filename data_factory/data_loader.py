import os
import pywt
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            train_series = np.float32(self.train[index:index + self.win_size])
            train_label = np.float32(self.test_labels[0:self.win_size])
            train_freq = abs(np.fft.fft(train_series))
            return train_series, train_freq, train_label

        elif (self.mode == 'val'):
            val_series = np.float32(self.val[index:index + self.win_size])
            val_label = np.float32(self.test_labels[0:self.win_size])
            val_freq = abs(np.fft.fft(val_series))
            return val_series, val_freq, val_label

        elif (self.mode == 'test'):
            test_series = np.float32(self.test[index:index + self.win_size])
            test_label = np.float32(self.test_labels[index:index + self.win_size])
            test_freq = abs(np.fft.fft(test_series))
            return test_series, test_freq, test_label
        else:
            data_series = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_label = np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_freq = abs(np.fft.fft(data_series))
            return data_series, data_freq, data_label


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            train_series = np.float32(self.train[index:index + self.win_size])
            train_label = np.float32(self.test_labels[0:self.win_size])
            train_freq = abs(np.fft.fft(train_series))
            return train_series, train_freq, train_label

        elif (self.mode == 'val'):
            val_series = np.float32(self.val[index:index + self.win_size])
            val_label = np.float32(self.test_labels[0:self.win_size])
            val_freq = np.fft.fft(val_series)
            return val_series, val_freq, val_label
        elif (self.mode == 'test'):
            test_series = np.float32(self.test[index:index + self.win_size])
            test_label = np.float32(self.test_labels[index:index + self.win_size])
            test_freq = abs(np.fft.fft(test_series))
            return test_series, test_freq, test_label
        else:
            data_series = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_freq = abs(np.fft.fft(data_series))
            return data_series, data_freq, data_label


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            train_series = np.float32(self.train[index:index + self.win_size])
            train_label = np.float32(self.test_labels[0:self.win_size])
            train_freq = abs(np.fft.fft(train_series))
            return train_series, train_freq, train_label

        elif (self.mode == 'val'):
            val_series = np.float32(self.val[index:index + self.win_size])
            val_label = np.float32(self.test_labels[0:self.win_size])
            val_freq = np.fft.fft(val_series)
            return val_series, val_freq, val_label
        elif (self.mode == 'test'):
            test_series = np.float32(self.test[index:index + self.win_size])
            test_label = np.float32(self.test_labels[index:index + self.win_size])
            test_freq = abs(np.fft.fft(test_series))
            return test_series, test_freq, test_label
        else:
            data_series = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_freq = abs(np.fft.fft(data_series))
            return data_series, data_freq, data_label

class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            train_series = np.float32(self.train[index:index + self.win_size])
            train_label = np.float32(self.test_labels[0:self.win_size])
            train_freq = abs(np.fft.fft(train_series))
            return train_series, train_freq, train_label

        elif (self.mode == 'val'):
            val_series = np.float32(self.val[index:index + self.win_size])
            val_label = np.float32(self.test_labels[0:self.win_size])
            val_freq = np.fft.fft(val_series)
            return val_series, val_freq, val_label
        elif (self.mode == 'test'):
            test_series = np.float32(self.test[index:index + self.win_size])
            test_label = np.float32(self.test_labels[index:index + self.win_size])
            test_freq = abs(np.fft.fft(test_series))
            return test_series, test_freq, test_label
        else:
            data_series = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            data_freq = abs(np.fft.fft(data_series))
            return data_series, data_freq, data_label


class SKABSegLoader(object):
    def __init__(self, data_path, win_size, step, mode='train'):
        self.mode = mode
        self.win_size = win_size
        self.step = step
        self.scaler = StandardScaler()
        all_files=[]
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
        
        for file in all_files:
            if 'anomaly-free' not in file:
                df = pd.read_csv(file, sep=';', index_col='datetime', parse_dates=True)
                features = df.drop(['anomaly', 'changepoint'], axis=1)
                label = df['anomaly']
                

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


def get_loader_dist(args, step=100, mode='train'):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if (args.dataset == 'SMD'):
        dataset = SMDSegLoader(args.data_path, args.win_size, step, mode)
    elif (args.dataset == 'MSL'):
        dataset = MSLSegLoader(args.data_path, args.win_size, 1, mode)
    elif (args.dataset == 'SMAP'):
        dataset = SMAPSegLoader(args.data_path, args.win_size, 1, mode)
    elif (args.dataset == 'PSM'):
        dataset = PSMSegLoader(args.data_path, args.win_size, 1, mode)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if mode == 'train':
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True)
    return dataloader


if __name__ == "__main__":
    SKAB_loader = SKABSegLoader(data_path='dataset/SKAB/data', win_size=100, step=1, mode='train')