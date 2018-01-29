import numpy as np
import scipy.integrate as integration
import matplotlib.pyplot as plt

def C(x):
    return np.cos((np.pi * (x ** 2))/2)

def S(x):
    return np.sin((np.pi * (x ** 2))/2)

def integrate(f, x_lower, x_upper):
    return integration.quad(f, x_lower, x_upper)


points_to_plot = []
for u1 in np.linspace(-8, 8, 3000):
    c = integrate(C, 0, u1)[0]
    s = integrate(S, 0, u1)[0]
    points_to_plot.append([c,s])

c_points = [l[0] for l in points_to_plot]
s_points = [l[1] for l in points_to_plot]

fig1, ax1 = plt.subplots()
ax1.plot(c_points, s_points, color = 'blue', lw = 0.5)
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))   # set position of y spine to y=0
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.set_xlabel("C(u)")
ax1.xaxis.set_label_coords(1.0, 0.55)
ax1.set_ylabel("S(u)", rotation='horizontal')
ax1.yaxis.set_label_coords(0.45, 0.95)
plt.title("The Cornu Spiral")
fig1.savefig("Core2.pdf")

#########

d = 10
lamb = 1
lengths_D_to_try = [30,50,100]


for D in lengths_D_to_try:
    amplitudes_to_plot = []
    for u2 in np.linspace(-25, 25, 5000):
        x1 = ((d/2) - u2) * np.sqrt(2/(lamb * D))
        x0 = (-(d/2) - u2) * np.sqrt(2/(lamb * D))
        c = integrate(C, x0, x1)[0]
        s = integrate(S, x0, x1)[0]
        intensity = c ** 2 + s ** 2
        amplitudes_to_plot.append([u2,intensity])

    positions = [l[0] for l in amplitudes_to_plot]
    intensities = [l[1] for l in amplitudes_to_plot]

    fig2, ax2 = plt.subplots()
    ax2.plot(positions, intensities, color = 'blue', lw = 0.5)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('data',0))   # set position of y spine to y=0
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.set_xlabel("Distance from centre of slit on plane normal to slit / cm")
    #ax2.xaxis.set_label_coords(1.0, 0.55)
    ax2.set_ylabel("Intensity / arb. units", rotation='horizontal')
    ax2.yaxis.set_label_coords(0.66, 0.95)
    plt.title("Diffraction Pattern for D = " + str(D) + "cm")
    fig2.savefig("Supplementary2_" + str(D) + ".pdf")