import numpy as np
import scipy.integrate as integration
import matplotlib.pyplot as plt

def C(x):
    '''
        Takes in argument x
        Returns integrand value for C(x)
    '''
    return np.cos((np.pi * (x ** 2))/2)

def S(x):
    '''
        Takes in argument x
        Returns integrand value for S(x)
    '''
    return np.sin((np.pi * (x ** 2))/2)

def integrate(f, x_lower, x_upper):
    '''
        Takes in (function, lower integration limit, upper integration limit)
        Returns accurately integrated result
    '''
    return integration.quad(f, x_lower, x_upper)[0]


points_to_plot = []
for u1 in np.linspace(-8, 8, 3000):
    c = integrate(C, 0, u1)
    s = integrate(S, 0, u1)
    points_to_plot.append([c,s])

c_points = [l[0] for l in points_to_plot]
s_points = [l[1] for l in points_to_plot]

fig1, ax1 = plt.subplots()
ax1.plot(c_points, s_points, color = 'blue', lw = 0.5)
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0)) # set position of x spine to x = 0
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0)) # set position of y spine to y = 0
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.set_xlim([-0.80, 0.801])
ax1.xaxis.set_ticks(np.arange(-0.80, 0.81, 0.20))
ax1.set_ylim([-0.80, 0.801])
ax1.yaxis.set_ticks(np.arange(-0.80, 0.81, 0.20))
ax1.set_xlabel("C(u)")
ax1.xaxis.set_label_coords(1.0, 0.55)
ax1.set_ylabel("S(u)", rotation='horizontal')
ax1.yaxis.set_label_coords(0.55, 0.95)
ax1.set_title("The Cornu Spiral")
ax1.set_aspect("equal")
fig1.savefig("Core2.pdf", bbox_inches = 'tight')

#########

d = 10
lamb = 1
lengths_D_to_try = [30,50,100]
tol = 1.5 # tolerance value to prevent discontinuous phase plot

for D in lengths_D_to_try:
    positions_to_plot = np.linspace(-25, 25, 3000)
    intensities_to_plot = np.array([])
    phases_to_plot = np.array([])
    for u2 in positions_to_plot:
        x1 = ((d/2) - u2) * np.sqrt(2/(lamb * D)) # lower integration limit
        x0 = (-(d/2) - u2) * np.sqrt(2/(lamb * D)) # upper integration limit
        c = integrate(C, x0, x1)[0]
        s = integrate(S, x0, x1)[0]
        intensity = c ** 2 + s ** 2
        phase = np.arctan(s/c)
        intensities_to_plot = np.append(intensities_to_plot, intensity)
        phases_to_plot = np.append(phases_to_plot, phase)

    phases_to_plot[phases_to_plot > tol] = np.nan
    intensities_to_plot = intensities_to_plot/max(intensities_to_plot)

    figs, axarr = plt.subplots(2, 1, figsize = (10,15))

    axarr[0].plot(positions_to_plot, intensities_to_plot, color='red', lw=0.5)
    axarr[0].xaxis.set_ticks_position('bottom')
    axarr[0].yaxis.set_ticks_position('left')
    axarr[0].spines['left'].set_position(('data', 0))
    axarr[0].spines['right'].set_color('none')
    axarr[0].spines['top'].set_color('none')
    axarr[0].set_xlabel("Distance from optical axis / cm")
    axarr[0].set_ylabel("Relative Intensity / arb. units", rotation='horizontal')
    axarr[0].yaxis.set_label_coords(0.65, 0.95)
    axarr[0].set_title("Diffraction Pattern for D = " + str(D) + "cm")
    axarr[0].set_aspect("auto")

    axarr[1].plot(positions_to_plot, phases_to_plot, color='blue', lw=0.5)
    axarr[1].set_ylim([-1.5, 1.51])
    axarr[1].yaxis.set_ticks(np.arange(-1.5, 1.51, 0.3))
    axarr[1].set_xlabel("Distance from optical axis / cm")
    axarr[1].set_ylabel("Relative Phase / rad")
    axarr[1].set_title("Phase of Diffraction Pattern for D = " + str(D) + "cm")
    axarr[1].set_aspect("auto")

    figs.savefig("Supplementary2_" + str(D) + ".pdf", bbox_inches = 'tight')