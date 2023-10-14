import numpy as np
from itertools import zip_longest
import csv

primary_sample_num = list(np.loadtxt("2023_10_04_pm_sample.txt",
                                     unpack=True, usecols=0, skiprows=2))
# print(primary_sample_num)

sample_num_counts = list(np.loadtxt("2023_10_04_pm_sample.txt",
                                    unpack=True, usecols=1, skiprows=2))
# print(sample_num_counts)

background_sample_num = list(np.loadtxt("2023_10_04_pm_background.txt",
                                        unpack=True, usecols=0, skiprows=2))
# print(background_sample_num)

background_num_counts = list(np.loadtxt("2023_10_04_pm_background.txt",
                                        unpack=True, usecols=1, skiprows=2))

data = [primary_sample_num, sample_num_counts]
columns_data = zip_longest(*data)

with open("2023_10_04_pm_sample.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(columns_data)
