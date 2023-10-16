import matplotlib.pyplot as plt

from Part_Two_Plate import *

if __name__ == "__main__":
    sample_num, num_of_counts = np.loadtxt(os.curdir + "/2022_10_05_pm_background.txt", skiprows=2, unpack=True)
    sorted_num_of_counts = np.sort(num_of_counts)
    range_1 = (min(num_of_counts), max(num_of_counts))

    plt.figure("Histogram")
    plt.title("Probability Density for each Number of Counts \nfrom The Background Radiation in One Sample")
    histogram_plot(num_of_counts, range_1, 1)
    print(num_of_counts.astype(int))
    average_num_of_counts = np.average(num_of_counts)
    poisson_probabilities = stats.poisson.pmf(sorted_num_of_counts, average_num_of_counts)
    print(poisson_probabilities)
    plt.plot(sorted_num_of_counts, poisson_probabilities, "yx-", label="Poisson distribution")
    gaussian_x = np.arange(range_1[0], range_1[1], 0.1)
    gaussian_probabilities = stats.norm.pdf(gaussian_x, average_num_of_counts, np.sqrt(average_num_of_counts))
    plt.plot(gaussian_x, gaussian_probabilities, "--", label="Gaussian distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_two_analysis_background")

    plt.figure("Plate Residual")
    plt.xlabel("number of counts")
    plt.ylabel("probability difference")
    plot_residual(sorted_num_of_counts, poisson_probabilities, stats.norm.pdf(sorted_num_of_counts,
                                                                              average_num_of_counts,
                                                                              np.sqrt(average_num_of_counts)))
    plt.title("Residual Plot Showing Difference between Poisson Distribution\n and Gaussian Distribution at each number"
              " of counts from\n Background Radiation")
    plt.tight_layout()
    plt.savefig("part_two_residual_background")
    plt.show()
