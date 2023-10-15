import numpy as np
from itertools import zip_longest
import csv

def float2int(lst):

    for i in np.arange(len(lst)):
        int_i = int(lst[i])
        lst.pop(i)
        lst.insert(i, int_i)
    return lst


primary_sample_num = list(np.loadtxt("2023_10_04_pm_sample.txt",
                                     unpack=True, usecols=0, skiprows=2))
float2int(primary_sample_num)
primary_sample_num.insert(0, "Sample Number")
# print(primary_sample_num)

sample_num_counts = list(np.loadtxt("2023_10_04_pm_sample.txt",
                                    unpack=True, usecols=1, skiprows=2))
float2int(sample_num_counts)
sample_num_counts.insert(0, "Number of Counts")
# print(sample_num_counts)

background_sample_num = list(np.loadtxt("2023_10_04_pm_background.txt",
                                        unpack=True, usecols=0, skiprows=2))
float2int(background_sample_num)
# print(background_sample_num)

background_sample_num.insert(0, "Background Sample Number")

background_num_counts = list(np.loadtxt("2023_10_04_pm_background.txt",
                                        unpack=True, usecols=1, skiprows=2))
float2int(background_num_counts)
background_num_counts.insert(0, "Background Number of Counts")

# Plate Data

primary_sample_num_plate = list(np.loadtxt("2022_10_05_pm_plate.txt",
                                     unpack=True, usecols=0, skiprows=2))
float2int(primary_sample_num_plate)
primary_sample_num_plate.insert(0, "Plate Sample Number")
# print(primary_sample_num)

sample_num_counts_plate = list(np.loadtxt("2022_10_05_pm_plate.txt",
                                    unpack=True, usecols=1, skiprows=2))
float2int(sample_num_counts_plate)
sample_num_counts_plate.insert(0, "Plate Number of Counts")
# print(sample_num_counts)

background_sample_num_plate = list(np.loadtxt("2022_10_05_pm_background.txt",
                                        unpack=True, usecols=0, skiprows=2))
# print(background_sample_num)
float2int(background_sample_num_plate)
background_sample_num_plate.insert(0, "Background Plate Sample Number")

background_num_counts_plate = list(np.loadtxt("2022_10_05_pm_background.txt",
                                        unpack=True, usecols=1, skiprows=2))
float2int(background_num_counts_plate)
background_num_counts_plate.insert(0, "Background Plate Number of Counts")


def create_csv(lst1, lst2, new_filename: str):

    data = [lst1, lst2]
    columns_data = zip_longest(*data)

    with open(new_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(columns_data)

    return 0


create_csv(primary_sample_num, sample_num_counts, "2023_10_04_pm_sample_1.csv")

create_csv(background_sample_num,
           background_num_counts, "2023_10_04_pm_background.csv")

create_csv(primary_sample_num_plate, sample_num_counts_plate,
           "2023_10_05_pm_plate.csv")

create_csv(background_sample_num_plate,
           background_num_counts_plate, "2023_10_05_pm_background_plate.csv")

