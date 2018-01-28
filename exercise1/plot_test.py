import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.sin(np.linspace(-10, 10, 100)))
plt.xlabel("Distance along axis (m)")
plt.ylabel("Distance perpendicular to axis (m)")
plt.title("Magnetic field strength")
plt.savefig("myplot.pdf")
