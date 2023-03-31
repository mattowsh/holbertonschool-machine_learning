#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# fruit is a matrix representing the number of fruit various people possess
# Define the necessary datasets:
owners = ["Farrah", "Fred", "Felicia"]
fruits = ["apples", "bananas", "oranges", "peaches"]
colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

# Define a column to create the fruit stack for each owner:
stack = np.zeros(len(owners))

# Create a bar, for any fruit, and append it to the stack:
for i in range(len(fruits)):
    plt.bar(owners, fruit[i], color=colors[i], width=0.5, bottom=stack)
    stack += fruit[i]

# Define plot settings:
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.ylim([0, 80])
plt.legend(fruits)

plt.show()
