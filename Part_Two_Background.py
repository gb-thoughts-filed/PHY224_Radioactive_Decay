from Part_Two_Plate import *

if __name__ == "__main__":
    sample_num, num_of_counts = np.loadtxt(os.curdir + "/2022_10_05_pm_background.txt", skiprows=2, unpack=True)
    sorted_num_of_counts = np.sort(num_of_counts)
    range_1 = (min(num_of_counts), max(num_of_counts))
    plt.figure("Histogram")
    histogram_plot(num_of_counts, range_1, 1)
    print(num_of_counts.astype(int))
    average_num_of_counts = np.average(num_of_counts)
    poisson_probabilities = stats.poisson.pmf(sorted_num_of_counts, average_num_of_counts)
    print(poisson_probabilities)
    plt.plot(sorted_num_of_counts, poisson_probabilities, "rx", label="Poisson distribution")
    gaussian_x = np.arange(range_1[0], range_1[1], 0.1)
    gaussian_probabilities = stats.norm.pdf(gaussian_x, average_num_of_counts, np.sqrt(average_num_of_counts))
    plt.plot(gaussian_x, gaussian_probabilities, "g-", label="Gaussian distribution")
    plt.show()
