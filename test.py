import torch
import torch.nn as nn
import pandas as pd

source_path = "dataset/PSM/train.csv"
data = pd.read_csv(source_path)
data = data.values[:, 1:]
features1 = data[0:100, :]
t = torch.tensor(features1)
t = t.unsqueeze(0).float()
t = t.permute([0, 2, 1])
conv1 = nn.Conv1d(in_channels=25, out_channels=150, kernel_size=3, padding=1, padding_mode='circular')

out = conv1(t)

print(out.permute([0, 2, 1]).shape)