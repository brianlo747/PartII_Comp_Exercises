import numpy as np
import matplotlib.pyplot as plt

def integrand(x):
    return (10**6) * np.sin(np.sum(x))

def monte_carlo(N):
    eight_d_points = np.array(np.random.rand(N, 8) * (np.pi/8)) #Generates 8-D random points within range
    integrand_of_points = np.apply_along_axis(integrand, 1, eight_d_points) #Calculates integrand for each of these points
    f_mean = (1/N) * np.sum(integrand_of_points)
    f_squared = (1/N) * np.sum(np.square(integrand_of_points))
    volume = ((np.pi/8)**8)

    integral_estimate = volume * f_mean
    error_estimate = volume * np.sqrt((f_squared - f_mean ** 2)/N)

    return [integral_estimate, error_estimate]

def multiple_runs(N_max, sep, n_t):
    list_of_Ns = np.array(np.logspace(2, N_max, num=sep)) #Creates logarithmacally spaced values from 2 to N_max
    list_of_Ns = list_of_Ns.astype(int) #and converts it into an integer list
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
            integrated_list = np.append(integrated_list,run[0])
        print(integrated_list)
        results.append([N, mean_accu/n_t, np.std(integrated_list), np.sqrt(error2_accu)/n_t])
        print("N = "+str(N)+" computation complete!")
    print(results)
    return results


results_table = multiple_runs(6, 25, 25)
values_of_N = [l[0] for l in results_table]
scatter_errors = [l[2] for l in results_table]
sigma_errors = [l[3] for l in results_table]

coeffs = np.polyfit(np.log(values_of_N), np.log(scatter_errors), deg=1)
poly = np.poly1d(coeffs)
yfit = lambda x: np.exp(poly(np.log(x)))

plt.loglog(values_of_N, scatter_errors, 'o', markersize=2, color='b',
           label="Standard Deviation of Monte-Carlo values")
plt.loglog(values_of_N, yfit(values_of_N), color='b',
           label = "Standard Deviation of Monte-Carlo values (Best-fit)")
plt.loglog(values_of_N, sigma_errors, color='o',
           label="Theoretical error")

plt.legend()
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Errors of the Monte-Carlo method against number of sample points N")
plt.savefig("Core1.pdf")