import os

import numpy
import numpy as np
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import warnings


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
    try:
        return np.exp(-average) * numpy.power(average, num_of_counts, dtype=np.float128) / \
            special.gamma(num_of_counts + 1)
    except AttributeError:
        warnings.filterwarnings("error")
        poisson_prob = []
        for i in num_of_counts:
            try:
                poisson_prob.append(np.exp(-average) * numpy.power(average, i) / \
                                    special.gamma(i + 1))
            except RuntimeWarning:
                poisson_prob.append(stats.poisson.pmf(np.round(i), average))
        return np.array(poisson_prob)

    # The power calculation maybe too big to calculate.In some cases this can be fixed by giving more memory to store
    # the result of the calculation. If the problem can be not resolved, the scipy poisson.pmf function will be used
    # with rounded N_s to avoid the power calculation


def chi_square(y, prediction, uncertainty, num_of_param):
    return numpy.sum(((y - prediction) / uncertainty) ** 2) / (len(y) - num_of_param)


if __name__ == "__main__":
    #
    sample_num_plate, num_of_counts_plate = np.loadtxt(os.curdir + "/2022_10_05_pm_plate.txt", skiprows=2, unpack=True)
    sample_num_background, num_of_counts_background = np.loadtxt(os.curdir + "/2022_10_05_pm_background.txt",
                                                                 skiprows=2, unpack=True)
    num_of_counts = (num_of_counts_plate - np.average(num_of_counts_background))
    sorted_num_of_counts = np.sort(num_of_counts)
    average_num_of_counts = np.average(num_of_counts)
    range_1 = (floor_dig(min(num_of_counts), 0), ceil_dig(max(num_of_counts), 0))
    plt.figure("Histogram")
    plt.title("probability density for number of counts in one sample")
    histogram_plot(num_of_counts, range_1, 1)
    print(num_of_counts, sorted_num_of_counts)
    print(average_num_of_counts)

    poisson_probabilities = poisson_distribution(sorted_num_of_counts, average_num_of_counts)
    print(poisson_probabilities)
    plt.plot(sorted_num_of_counts, poisson_probabilities, "gx", label="Poisson distribution")

    gaussian_x = np.arange(range_1[0], range_1[1], 0.1)
    gaussian_probabilities = stats.norm.pdf(gaussian_x, average_num_of_counts, np.sqrt(average_num_of_counts))
    plt.plot(gaussian_x, gaussian_probabilities, label="Gaussian distribution")
    plt.show()
