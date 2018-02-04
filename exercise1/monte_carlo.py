import numpy as np
import matplotlib.pyplot as plt


def integrand(x):
    '''
        Takes in a numpy array of x-values in 8 dimensions
        Returns integrand value
    '''
    return (10 ** 6) * np.sin(np.sum(x))


def monte_carlo(N):
    '''
        Takes in your desired number of sample points to be generated
        Returns Monte-Carlo estimate with theoretical error
    '''
    eight_d_points = np.array(np.random.rand(N, 8) * (np.pi / 8))  # Generates 8-D random points within range
    integrand_of_points = np.apply_along_axis(integrand, 1,
                                              eight_d_points)  # Calculates integrand for each of these points
    f_mean = (1 / N) * np.sum(integrand_of_points)
    f_squared = (1 / N) * np.sum(np.square(integrand_of_points))
    volume = ((np.pi / 8) ** 8)

    integral_estimate = volume * f_mean
    error_estimate = volume * np.sqrt((f_squared - f_mean ** 2) / N)

    return [integral_estimate, error_estimate]


def multiple_runs(N_max, sep, n_t):
    '''
        Takes in (Maximum number of sample points N_max, Different Ns to be tested from 2 to N_max,
        Number of times each N is tested)
        Returns a list of results containing lists in the form of
        (Sample points, Mean integral estimate, Overall Scatter, Overall Theoretical Error)
    '''
    list_of_Ns = np.array(np.logspace(2, N_max, num=sep))  # Creates logarithmacally spaced values from 2 to N_max
    list_of_Ns = list_of_Ns.astype(int)  # and converts it into an integer list
    print(list_of_Ns)
    results = []
    for N in list_of_Ns:
        mean_accu = 0
        error2_accu = 0
        integrated_list = np.array([])
        for i in range(n_t):
            run = monte_carlo(N)
            mean_accu += run[0]
            error2_accu += (run[1] ** 2)
            integrated_list = np.append(integrated_list, run[0])
        print(integrated_list)
        results.append([N, mean_accu / n_t, np.std(integrated_list) / np.sqrt(n_t), np.sqrt(error2_accu / n_t)])
        print("N = " + str(N) + " computation complete!")
    print(results)
    return results


results_table = multiple_runs(6, 25, 25)
values_of_N = [l[0] for l in results_table]
scatter_errors = [l[2] for l in results_table]
theoretical_errors = [l[3] for l in results_table]

coeffs_scatter = np.polyfit(np.log(values_of_N), np.log(scatter_errors), deg=1)  # Fits scatter error values
coeffs_theoretical = np.polyfit(np.log(values_of_N), np.log(theoretical_errors), deg=1)  # Fits theoretical error values
poly_scatter = np.poly1d(coeffs_scatter)
poly_theoretical = np.poly1d(coeffs_theoretical)
yfit_scatter = lambda x: np.exp(poly_scatter(np.log(x)))
yfit_theoretical = lambda x: np.exp(poly_theoretical(np.log(x)))

print(poly_scatter)
print(poly_theoretical)

plt.loglog(values_of_N, scatter_errors, 'o', markersize=2, color='blue',
           label="Standard Deviation of Monte-Carlo estimates")
plt.loglog(values_of_N, yfit_scatter(values_of_N), color='blue',
           label="Standard Deviation of Monte-Carlo estimates (Best-fit)")
plt.loglog(values_of_N, theoretical_errors, 'o', markersize=2, color='orange',
           label="Theoretical error of Monte-Carlo estimates")
plt.loglog(values_of_N, yfit_theoretical(values_of_N), color='orange',
           label="Theoretical error of Monte-Carlo estimates (Best-fit)")

plt.legend(prop={'size': 8})
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Errors of the Monte-Carlo method against number of sample points N")
plt.savefig("Core1.pdf")
