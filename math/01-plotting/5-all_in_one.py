#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# 1st plot: Line Graph
y0 = np.arange(0, 11) ** 3

# 2nd plot: Scatter plot
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

# 3rd plot: Line Graph logarithmically scaled
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

# 4th plot: Two Line Graph
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

# 5th plot: Histogram
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


# All-in-one graph:
plt.suptitle('All in One', fontsize=16)

# 1st plot: Line Graph
plt.subplot(3, 2, 1)
plt.xlim([0, 10])
plt.plot(y0, c='red')

# 2nd plot: Scatter plot
plt.subplot(3, 2, 2)
plt.title("Men's Height vs Weight", size="x-small")
plt.xlabel("Height (in)", size="x-small")
plt.ylabel("Weight (lbs)", size="x-small")
plt.scatter(x1, y1, c='magenta', s=10)

# 3rd plot: Line Graph logarithmically scaled
plt.subplot(3, 2, 3)
plt.title("Exponential Decay of C-14", size="x-small")
plt.xlabel("Time (years)", size="x-small")
plt.ylabel("Fraction Remaining", size="x-small")
plt.xlim([0, 28651])
plt.yscale("log")
plt.plot(x2, y2)

# 4th plot: Two Line Graph
plt.subplot(3, 2, 4)
plt.title("Exponential Decay of C-14", size="x-small")
plt.xlabel("Time (years)", size="x-small")
plt.ylabel("Fraction Remaining", size="x-small")
plt.xlim([0, 20001])
plt.ylim([0, 1])
plt.plot(x3, y31, label="C-14", c="red", linestyle='dashed')
plt.plot(x3, y32, label="Ra-226", c="green", linestyle='solid')
plt.legend()

# 5th plot: Histogram
plt.subplot(3, 2, (5, 6))
plt.title("Project A", size="x-small")
plt.xlabel("Grades", size="x-small")
plt.ylabel("Number of Students", size="x-small")
plt.xlim([0, 100])
plt.ylim([0, 30])
bins = list(range(0, 110, 10))
plt.xticks(bins)
plt.hist(student_grades, bins=bins, edgecolor="black")

# Automatically adjusts the spacing between subplots:
plt.tight_layout()
plt.show()
