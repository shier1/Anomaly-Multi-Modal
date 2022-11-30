import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt 

source_path = "dataset/PSM/train.csv"
data = pd.read_csv(source_path)
data = data.values[:, 1:]
features1 = data[0:100, 0]
ca, cd = pywt.dwt(features1, 'haar', 'sym')
# plt.plot(ca)
# plt.plot(cd)
plt.scatter(x=np.arange(len(features1)),y=features1)
plt.show()