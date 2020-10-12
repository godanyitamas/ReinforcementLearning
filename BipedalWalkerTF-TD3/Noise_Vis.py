""" Visualization of OU noise """
from OUnoise import OUNoise
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = 250
noise = OUNoise(size=(1,), seed=0, mu=0.1, theta=0.7, sigma=0.7)
noise_list_ou = []
noise_list_tf = []
for i in range(x):
    noise_list_ou.append(noise.sample().clip(-0.5, 0.5))
    noise_list_tf.append(tf.random.normal(shape=(1,), stddev=0.3, mean=0.0))

noise_list_ou = np.asarray(noise_list_ou).clip(-0.5, 0.5)
noise_list_tf = np.asarray(noise_list_tf).clip(-0.5, 0.5)

plt.style.use('seaborn')
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(noise_list_tf)
ax2.plot(noise_list_ou)
plt.show()
