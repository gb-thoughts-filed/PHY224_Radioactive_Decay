import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math


def linear_regression_model(x, a, b):
    return a*x + b


def nonlinear_regression_model(x, c, d):
    return d*math.e**(c*x)

