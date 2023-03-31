#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Set title and labels:
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

# Histogram:
# Setting axis:
plt.xlim([0, 100])
plt.ylim([0, 30])

# bins = intervals along the x-axis that are used to group the data
# edgecolor = border of each bar
# xtricks = set the tick locations and labels of the x-axis
bins = list(range(0, 110, 10))
plt.xticks(bins)
plt.hist(student_grades, bins=bins, edgecolor="black")

plt.show()
