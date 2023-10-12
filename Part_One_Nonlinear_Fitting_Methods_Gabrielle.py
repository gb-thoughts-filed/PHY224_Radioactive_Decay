# Instruction One: Import Required Functions and Modules

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math


# Instruction Two: Define the Model Functions


def linear_regression_model(x, a, b):
    return a*x + b


def nonlinear_regression_model(x, c, d):
    return d*math.e**(c*x)

# Instruction Three: Load the data using loadtxt()


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
# print(background_num_counts)

# Instruction Four: Subtract the mean background radiation from the data.

mean_bkgrnd_radiation = np.mean(background_num_counts)

# print(mean_bkgrnd_radiation)


def bkgrnd_rad_subtractor(lst_n_t: list, avg_bkgrnd_radiation: float) -> list:

    calculated_n_s = []

    for i in lst_n_t:
        calculated_n_s.append(i - avg_bkgrnd_radiation)

    return calculated_n_s


final_radiation_reading = bkgrnd_rad_subtractor(sample_num_counts,
                                                mean_bkgrnd_radiation)
# print(final_radiation_reading)

# Instruction Five: Calculate the uncertainty for each data point.


def uncertainty_calculation(sample_lst: list, bkgrnd_mean: float):

    uncertainties_lst = []

    for i in sample_lst:
        uncertainties_lst.append(np.sqrt(i + bkgrnd_mean))

    return uncertainties_lst


sample_uncertainties = uncertainty_calculation(sample_num_counts,
                                               mean_bkgrnd_radiation)

# print(sample_uncertainties)

# Instruction Six: Convert the count data into rates.

disp_t = 20.0

# displacement given on lab Quercus page.


def radiation_rate_calc(samples: list, uncertainties: list,
                        time_displacement: float):

    lst_rates = []

    for i in np.arange(len(samples)):
        sample_rate = samples[i]/time_displacement
        uncertainty_rate = uncertainties[i]/time_displacement
        lst_rates.append((sample_rate, uncertainty_rate))

    return lst_rates


calculated_rates = radiation_rate_calc(
    final_radiation_reading, sample_uncertainties, disp_t)

# print(calculated_rates)

# Instruction Seven: Perform the linear regression on (x_i, log(y_i))
# using f as the model function

# Getting Time Data

time_data = []
for i in primary_sample_num:
    time_data.append((i-1)*20/60)

# Getting log of y_i


def unpack_calculated_rates(calc_rates: list):
    lst1 = []
    lst2 = []
    for i in calc_rates:
        lst1.append(i[0])
        lst2.append(i[1])
    return lst1, lst2


ydata, uncertainty_ydata = unpack_calculated_rates(calculated_rates)

# print(ydata)
# print(uncertainty_ydata)

# Linear Fit

logged_ydata = np.log(ydata)
logged_uncertainty_ydata = []

for i in np.arange(len(uncertainty_ydata)):
    logged_uncertainty_ydata.append(uncertainty_ydata[i]/ydata[i])

# print(logged_ydata)
# print(logged_uncertainty_ydata)

popt_log_linear, pcov_log_linear = scipy.optimize.curve_fit(
    linear_regression_model, np.array(time_data), logged_ydata,
    sigma=logged_uncertainty_ydata, absolute_sigma=True)

print(popt_log_linear)
print(pcov_log_linear)

# print(time_data)

# Nonlinear Fit

popt_nonlinear, pcov_nonlinear = scipy.optimize.curve_fit(
    nonlinear_regression_model, time_data, ydata,
    sigma=uncertainty_ydata, absolute_sigma=True)

print(popt_nonlinear)
print(pcov_nonlinear)

# Theoretical Fit

y_theo_data = []

for i in np.arange(len(time_data)):
    y_i = (ydata[0])*(0.5**(time_data[i]/2.6))
    y_theo_data.append(y_i)

plt.figure(1)
plt.errorbar(time_data, ydata,
             yerr=uncertainty_ydata, marker="o", ls='',
             label="Raw Data")

plt.plot(time_data, math.e**linear_regression_model(
    np.array(time_data), *popt_log_linear), ls='--',
         label='e^(Linear Regression Model)')

plt.plot(time_data,
         nonlinear_regression_model(np.array(time_data), *popt_nonlinear),
         ls="-", label='Nonlinear Regression Model')
plt.plot(time_data, y_theo_data, ls='-.', label='Theoretical Model')

# plt.yscale('log')

plt.show()

# Instruction 8: Calculate and output
# half-life values for each regression method.

# Linear Regression Half Life Calculation

t_half_linear = (np.log(1/2))/popt_log_linear[0]
print("This is the linear regression half-life ->")
print(t_half_linear)

# Nonlinear Regression Half Life Calculation

t_half_nonlinear = (np.log(1/2))/popt_nonlinear[0]
print("This is the nonlinear regression half-life ->")
print(t_half_nonlinear)

# Instruction 9: Calculate the variance.


