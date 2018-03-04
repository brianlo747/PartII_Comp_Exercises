#####################
# Importing Packages
#####################

import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt

#####################
# Helper Functions
#####################

R = 1
I = 1/const.mu_0

def generate_N_coils(N, n, D):
    """

    :param N: (A positive integer) Number of coils in system
    :param n: (A positive integer) Number of coil elements in each coil
    :return: (An array) Rows containing x,y,z positions of coil elements and angle from z=0 axis
    """

    tangent_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if N == 1:
        coils_x = np.repeat(0., n)
    else:
        coils_x = np.repeat(np.linspace(-D/2, D/2, N), n)

    coils_y = np.tile(R * np.sin(tangent_angles),N)
    coils_z = np.tile(R * np.cos(tangent_angles),N)

    element_length = 2 * R * np.pi / n

    return np.dstack((coils_x,coils_y,coils_z,np.tile(tangent_angles,N)))[0], element_length

def dB_contribution(I,dl,r):
    """

    :param I: (A number) Current carried by wire
    :param dl: (A 3D vector) Wire length element
    :param r: (A 3D vector) Location with respect to current element
    :return: (A 3D vector) dB calculated in equation (14) from Biot-Savart Law
    """
    return -((const.mu_0 * I) / (4*np.pi)) * (np.cross(dl,r)/(np.linalg.norm(r) ** 3))

def index_to_coord_mapping(array_index, array_sep, array_min):
    return array_min + array_index * array_sep

def calculate_B_total(config_array):
    """

    :param element_list:
    :param config_array:
    :return:
    """
    x_min = config_array[0, 0]
    x_max = config_array[0, 1]
    x_num = config_array[0, 2]

    y_min = config_array[1, 0]
    y_max = config_array[1, 1]
    y_num = config_array[1, 2]

    z_min = config_array[2, 0]
    z_max = config_array[2, 1]
    z_num = config_array[2, 2]


    if x_num > 1:
        x_sep = (x_max - x_min) / (x_num - 1)
    else:
        x_sep = 0

    if y_num > 1:
        y_sep = (y_max - y_min) / (y_num - 1)
    else:
        y_sep = 0

    if z_num > 1:
        z_sep = (z_max - z_min) / (z_num - 1)
    else:
        z_sep = 0

    field = np.zeros([np.int(z_num),np.int(y_num),np.int(x_num),3])
    element_list, element_length = generate_N_coils(N, n, D)

    for element in element_list:
        for z_index in range(len(field)):
            for y_index in range(len(field[z_index,:,:])):
                for x_index in range(len(field[z_index,y_index,:])):

                    x_point = index_to_coord_mapping(x_index, x_sep, x_min)
                    y_point = index_to_coord_mapping(y_index, y_sep, y_min)
                    z_point = index_to_coord_mapping(z_index, z_sep, z_min)

                    x_wire = element[0]
                    y_wire = element[1]
                    z_wire = element[2]

                    dl = np.array([0, element_length * np.cos(element[3]),
                                   - element_length * np.sin(element[3])])
                    r = np.array([x_point - x_wire, y_point - y_wire, z_point - z_wire])
                    field[z_index, y_index, x_index] = field[z_index, y_index, x_index] + dB_contribution(I,dl,r)

    return field

def plot_2d(field, name):
    x, y = np.meshgrid(np.linspace(config[0, 0], config[0, 1], config[0, 2]),
                       np.linspace(config[1, 0], config[1, 1], config[1, 2]))
    vx = field[:, :, 0]
    vy = field[:, :, 1]

    fig, ax = plt.subplots()
    if N == 1:
        coils_x = np.array([0.])
    else:
        coils_x = np.linspace(-D/2, D/2, N)
    coils_y_pos = np.repeat(+R, coils_x.size)
    coils_y_neg = np.repeat(-R, coils_x.size)
    plt.plot(coils_x, coils_y_pos, 'o', color = "red")
    plt.plot(coils_x, coils_y_neg, 'o', color = "red")
    plt.quiver(x, y, vx, vy, pivot='middle', headwidth=4, headlength=6)
    fig.savefig(name, bbox_inches='tight')

def plot_2d_vicinity(field, name):
    x, y = np.meshgrid(np.linspace(config[0, 0], config[0, 1], config[0, 2]),
                       np.linspace(config[1, 0], config[1, 1], config[1, 2]))
    vx = field[:, :, 0]
    vy = field[:, :, 1]

    fig, ax = plt.subplots()
    plt.quiver(x, y, vx, vy, pivot='middle', headwidth=4, headlength=6)
    fig.savefig(name, bbox_inches='tight')

def plot_contour(field, central_magnitude, name):
    x, y = np.meshgrid(np.linspace(config[0, 0], config[0, 1], config[0, 2]),
                       np.linspace(config[1, 0], config[1, 1], config[1, 2]))
    resultant_field_magnitude = np.linalg.norm(field, axis=2)
    magnitude_difference = resultant_field_magnitude - central_magnitude / central_magnitude
    fig, ax = plt.subplots()
    plt.contourf(x, y, magnitude_difference)
    fig.savefig(name, bbox_inches='tight')


## Core 1

N = 1
n = 32
D = 0

config = np.array([[-1,1,100],[0,0,1],[0,0,1]])
resultant_field = calculate_B_total(config)[0,0,:]

resultant_field_magnitude = np.linalg.norm(resultant_field, axis=1)
x = np.linspace(config[0, 0],config[0, 1],config[0, 2])
B_theoretical = const.mu_0 * I * R ** 2 / (2 * (R ** 2 + x ** 2) ** (3/2))

fig1, ax1 = plt.subplots()
ax1.plot(x,resultant_field_magnitude, lw = 1, color = 'blue')
ax1.plot(x,B_theoretical, lw = 0.5, color = 'red')
ax1.set_xlabel("x / m")
ax1.set_ylabel("Magnetic Field B / T")
ax1.set_title("Magnetic Field Strength along x-axis for a single coil")
fig1.savefig("Core1-Magnetic-x.pdf", bbox_inches = 'tight')

fig2, ax2 = plt.subplots()
ax2.plot(x,resultant_field_magnitude - B_theoretical, lw = 1, color = 'blue')
ax2.set_xlabel("x / m")
ax2.set_ylabel("Magnetic Field Difference B / T")
ax2.set_title("Magnetic Field Strength difference between model and theory along x-axis for a single coil")
fig2.savefig("Core1-Magnetic-diff-x.pdf", bbox_inches = 'tight')

config = np.array([[-1, 1, 20], [-0.8, 0.8, 20], [0, 0, 1]])
resultant_field = calculate_B_total(config)[0, :, :]
plot_2d(resultant_field, "Core1-vector-field.pdf")

## Core 2

N = 2
n = 32
D = 1

config = np.array([[-1, 1, 20], [-0.8, 0.8, 20], [0, 0, 1]])
resultant_field = calculate_B_total(config)[0, :, :]
plot_2d(resultant_field, "Core2-vector-field.pdf")

#config = np.array([[0,0,1],[0,0,1],[0,0,1]])
#central_magnitude = np.linalg.norm(calculate_B_total(config)[0,0,0])
central_magnitude = (4/5) ** (3/2) * const.mu_0 * I / R

config = np.array([[-0.05,0.05,20],[-0.05,0.05,20],[0,0,1]])
resultant_field = calculate_B_total(config)[0,:,:]
plot_2d_vicinity(resultant_field, "Core2-vector-field-vicinity.pdf")
plot_contour(resultant_field, central_magnitude, "Core2-contour.pdf")

## Supplementary

N = 20
n = 32
D = 10

config = np.array([[-6, 6, 20], [-0.75, 0.75, 20], [0, 0, 1]])
resultant_field = calculate_B_total(config)[0, :, :]
plot_2d(resultant_field, "Supp-vector-field.pdf")

#config = np.array([[0,0,1],[0,0,1],[0,0,1]])
#central_magnitude = np.linalg.norm(calculate_B_total(config)[0,0,0])
central_magnitude = (4/5) ** (3/2) * const.mu_0 * I / R

config = np.array([[-0.05,0.05,50],[-0.05,0.05,50],[0,0,1]])
resultant_field = calculate_B_total(config)[0,:,:]
#plot_2d(resultant_field, "Supp-vector-field-vicinity.pdf")
plot_contour(resultant_field, central_magnitude, "Supp-contour.pdf")