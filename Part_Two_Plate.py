import os

import numpy
import numpy as np
from scipy import stats
from scipy import special
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


def poisson_distribution(num_of_counts, average):
    return np.exp(-average) * numpy.power(average, np.longdouble(num_of_counts), dtype=np.float128) / special.gamma(num_of_counts + 1)
    # dtype=np.float128 added for bigger power calculations.
    # If AttributeError: module 'numpy' has no attribute 'float128' is returned, change it to dtype=np.float64


if __name__ == "__main__":
    sample_num_plate, num_of_counts_plate = np.loadtxt(os.curdir + "/2022_10_05_pm_plate.txt", skiprows=2, unpack=True)
    sample_num_background, num_of_counts_background = np.loadtxt(os.curdir + "/2022_10_05_pm_background.txt",
                                                                 skiprows=2, unpack=True)
    count_rate = (num_of_counts_plate - np.average(num_of_counts_background))  # delta t given by data
    # description
    sorted_count_rate = np.sort(count_rate)
    average_count_rate = np.average(count_rate)
    range_1 = (floor_dig(min(count_rate), 0), ceil_dig(max(count_rate), 0))
    plt.figure("Histogram")
    histogram_plot(count_rate, range_1, 1)
    print(count_rate, sorted_count_rate)
    print(average_count_rate)

    poisson_probabilities = poisson_distribution(sorted_count_rate, average_count_rate)
    plt.plot(sorted_count_rate, poisson_probabilities, "gx", label="Poisson distribution")

    gaussian_x = np.arange(range_1[0], range_1[1], 0.1)
    gaussian_probabilities = stats.norm.pdf(gaussian_x, average_count_rate, np.sqrt(average_count_rate))
    plt.plot(gaussian_x, gaussian_probabilities, label="Gaussian distribution")
    plt.show()
