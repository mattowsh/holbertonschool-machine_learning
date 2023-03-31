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
# bins = qty of columns
# edgecolor = border of each bar
plt.xlim([0, 101])
plt.ylim([0, 31])
plt.hist(student_grades, bins=6, edgecolor="black")

plt.show()
