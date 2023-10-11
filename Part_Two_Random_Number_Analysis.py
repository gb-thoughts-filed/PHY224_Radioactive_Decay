import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

from scipy.stats import poisson


def ceil_dig(num, decimal):
    return np.ceil(num * (10 ** decimal)) * 10 ** (-decimal)


def floor_dig(num, decimal):
    return np.floor(num * (10 ** decimal)) * 10 ** (-decimal)


def histogram_plot(data, rounded_range):
    plt.xlabel("Number of Counts")
    plt.ylabel("Frequency Density")
    # round the range to nearest 10 that in include the all the num in data.
    print(rounded_range, (rounded_range[1] - rounded_range[0]) / 5)
    plt.hist(data, bins=int((rounded_range[1] - rounded_range[0]) / 5), density=True, stacked=True, range=rounded_range
             , edgecolor='black', linewidth=1.2)


if __name__ == "__main__":
    sample_num, num_of_counts = np.loadtxt(os.curdir + "/2022_10_05_pm_plate.txt", skiprows=2, unpack=True)
    sorted_num_of_counts = np.sort(num_of_counts)
    range_10 = (floor_dig(min(num_of_counts), -1), ceil_dig(max(num_of_counts), -1))
    plt.figure("Histogram")
    histogram_plot(num_of_counts, range_10)
    print(num_of_counts.astype(int))
    probabilities = poisson.pmf(sorted_num_of_counts, np.average(num_of_counts))
    print(probabilities)
    plt.plot(sorted_num_of_counts, probabilities)
    plt.show()
