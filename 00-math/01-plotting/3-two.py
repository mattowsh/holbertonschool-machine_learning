#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Set title and labels:
plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")

# Line graph:
plt.xlim([0, 20001])
plt.ylim([0, 1])

# Setting the curves and legends:
plt.plot(x, y1, label="C-14", c="red", linestyle='dashed')
plt.plot(x, y2, label="Ra-226", c="green", linestyle='solid')
plt.legend()

plt.show()
