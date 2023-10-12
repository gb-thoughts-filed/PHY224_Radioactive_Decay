import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def ceil_dig(num, decimal):
    return np.ceil(num * (10 ** decimal)) * 10 ** (-decimal)


def floor_dig(num, decimal):
    return np.floor(num * (10 ** decimal)) * 10 ** (-decimal)


def histogram_plot(data, rounded_range, interval):
    plt.xlabel("Number of Counts")
    plt.ylabel("Frequency Density")
    # round the range to nearest 10 that in include the all the num in data.
    print(rounded_range, (rounded_range[1] - rounded_range[0]) / interval)
    plt.hist(data, bins=int((rounded_range[1] - rounded_range[0]) / interval), density=True,
             range=rounded_range
             , edgecolor='black', linewidth=1.2)


if __name__ == "__main__":
    sample_num_plate, num_of_counts_plate = np.loadtxt(os.curdir + "/2022_10_05_pm_plate.txt", skiprows=2, unpack=True)
    sample_num_background, num_of_counts_background = np.loadtxt(os.curdir + "/2022_10_05_pm_background.txt",
                                                                 skiprows=2, unpack=True)
    count_rate = (num_of_counts_plate - np.average(num_of_counts_background)) / 5  # delta t given by data description
    sorted_count_rate = np.sort(count_rate)
    range_1 = (np.floor(min(count_rate)), np.ceil(max(count_rate)))
    plt.figure("Histogram")
    histogram_plot(count_rate, range_1, 1)
    print(count_rate, sorted_count_rate)
    average_num_of_counts = np.average(count_rate)
    print(average_num_of_counts)

    poisson_probabilities = stats.poisson.pmf(np.round(sorted_count_rate), np.round(average_num_of_counts))
    print(poisson_probabilities)
    plt.plot(sorted_count_rate, poisson_probabilities, "rx", label="Poisson distribution")

    poisson_probabilities_num = stats.poisson.pmf(np.sort(num_of_counts_plate), np.average(num_of_counts_plate))
    print(poisson_probabilities_num)
    plt.plot(sorted_count_rate, poisson_probabilities_num, "gx", label="Poisson distribution")

    gaussian_x = np.arange(range_1[0], range_1[1], 0.1)
    gaussian_probabilities = stats.norm.pdf(gaussian_x, average_num_of_counts, np.sqrt(average_num_of_counts))
    plt.plot(gaussian_x, gaussian_probabilities, label="Gaussian distribution")
    plt.show()
