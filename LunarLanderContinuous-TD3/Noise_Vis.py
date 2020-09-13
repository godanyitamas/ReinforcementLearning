""" Visualization of OU noise """
from OUnoise import OUNoise
import numpy as np
import matplotlib.pyplot as plt

x = 250
noise = OUNoise(size=(1, 4), seed=0, mu=0, theta=0.3, sigma=0.4)
noise_list = []
for i in range(x):
    noise_list.append(noise.sample())

noise_list = np.reshape(noise_list, newshape=(x, 4))
# noise_list = noise_list.clip(-1, 1)
# print(np.shape(noise_list))
# print(noise_list[:, 1])
plt.style.use('seaborn')
plt.plot(noise_list[:, 1])
# plt.plot(noise_list[:, 2])
plt.show()
