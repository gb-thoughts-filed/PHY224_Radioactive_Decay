# Instruction One: Import Required Functions and Modules
from Part_Two_Plate import chi_square
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math


# Instruction Two: Define the Model Functions


def linear_regression_model(x, a, b):
    return a*x + b


def nonlinear_regression_model(x, c, d):
    return d*np.e**(c*x)

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

print(mean_bkgrnd_radiation)


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

plt.plot(time_data, np.e**linear_regression_model(
    np.array(time_data), *popt_log_linear), ls='--',
         label='e^(Linear Regression Model)')

plt.plot(time_data,
         nonlinear_regression_model(np.array(time_data), *popt_nonlinear),
         ls="-", label='Nonlinear Regression Model')
plt.plot(time_data, y_theo_data, ls='-.', label='Theoretical Model')

plt.xlabel("Time (t) in Minutes")
plt.ylabel("Ln(Count Rate in Counts per 20 Seconds)")
plt.title("Log Plot of Time in Minutes v.s. "
          "The Raw Count Rate, "
          "The Count Rate Predicted by a Linear Regression Model,"
          "The Count Rate Predicted by a Nonlinear Regression Model,"
          "The Count Rate Predicted by a Theoretical Model", wrap=True)

plt.legend()

# plt.yscale('log')

# plt.show()

# Residual Plot Calculations


def difference_calc(model_data, raw_data):
    lst = []
    for j in np.arange(len(raw_data)):
        lst.append(abs(model_data[j]-raw_data[j]))
    return lst


residual_linear = difference_calc(list(np.e**linear_regression_model(
    np.array(time_data),*popt_log_linear)), ydata)

residual_nonlinear = difference_calc(list(
    nonlinear_regression_model(np.array(time_data), *popt_nonlinear)), ydata)

residual_theoretical = difference_calc(list(y_theo_data), ydata)


plt.figure(2)

plt.plot(time_data, residual_linear, marker="X",
         label='e^(Linear Regression Model)')

plt.plot(time_data,
         residual_nonlinear,
         marker="8", label='Nonlinear Regression Model')
plt.plot(time_data, residual_theoretical, marker="D",
         label='Theoretical Model')

plt.xlabel("Time (t) in Minutes")
plt.ylabel("Model Count Rate in Counts per 20 Seconds - Measured Count Rate",
           wrap=True)
plt.title("Residual Plot of Time in Minutes v.s. "
          "The Count Rate Predicted by a Linear Regression Model,"
          "The Count Rate Predicted by a Nonlinear Regression Model,"
          "The Count Rate Predicted by a Theoretical Model"
          "All With the Raw Measured Count Rate Subtracted", wrap=True)

plt.legend()
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

# Linear Regression Standard Deviation

stddev_hlf_life_linear = (np.sqrt(
    pcov_log_linear[0, 0]))/(popt_log_linear[0])**2

print("\nThis is the standard deviation for the linear regression plot ->")
print(stddev_hlf_life_linear)
print("\nLinear low bound ->")
print(t_half_linear-stddev_hlf_life_linear)
print("Linear high bound ->")
print(t_half_linear+stddev_hlf_life_linear)

# Nonlinear Regression Standard Deviation
stddev_hlf_life_nonlinear = (np.sqrt(
    pcov_nonlinear[0, 0]))/(popt_nonlinear[0])**2

print("\nThis is the standard deviation for the nonlinear regression plot ->")
print(stddev_hlf_life_nonlinear)
print("\nNonlinear low bound ->")
print(t_half_nonlinear-stddev_hlf_life_linear)
print("Nonlinear high bound ->")
print(t_half_nonlinear+stddev_hlf_life_linear)

plt.figure(4)
plt.hlines(1, 2.48, 2.63)
plt.title("Comparison Between Linear & "
          "Non-Linear Regression Model Exponents and Std. Deviations",
          wrap=True)
lin_reg_pts = [2.5426157839996466, 2.585246536330504, 2.6278772886613617]
nonlin_reg_pts = [2.4859320237955727,	2.52856277612643, 2.5711935284572878]
theo_pts = [2.6]
plt.eventplot(lin_reg_pts, orientation='horizontal', colors="r",
              label="Linear Regression Model: Half-Life,"
                    " Half-Life + Std. Dev., Half-Life - Std. Dev.")
plt.eventplot(nonlin_reg_pts, orientation='horizontal', colors="b",
              label="Nonlinear Regression Model: Half-Life,"
                    " Half-Life + Std. Dev., Half-Life - Std. Dev.")
plt.eventplot(theo_pts, orientation='horizontal', colors="g",
              label="Theoretical Model Half-Life")
plt.legend()
plt.axis()
plt.show()

chi_squared_linear = chi_square(
    ydata,
    np.e**linear_regression_model(np.array(time_data), *popt_log_linear),
    logged_uncertainty_ydata,
    2)

print("\n Chi-Squared Linear Calculation ->")
print(chi_squared_linear)

chi_squared_nonlinear = chi_square(
    ydata,
    nonlinear_regression_model(np.array(time_data), *popt_nonlinear),
    uncertainty_ydata,
    2)

print("\n Chi-Squared Nonlinear Calculation ->")
print(chi_squared_nonlinear)

chi_squared_theoretical = chi_square(
    ydata,
    np.array(y_theo_data),
    uncertainty_ydata,
    2)

print("\n Chi-Squared Theoretical Calculation ->")
print(chi_squared_theoretical)
