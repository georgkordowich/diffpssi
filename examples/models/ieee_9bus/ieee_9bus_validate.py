"""
This file compare the results of the IEEE 9 bus model Python Simulation with the results of PowerFactory.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tools.colors import *

# load the data
path_data_mine = r'data/original_data_t.npy'
path_power_factory = r'data/pf_data.elm'
data_ours = np.load(path_data_mine)
data_pf = np.loadtxt(path_power_factory, skiprows=2)
length = len(data_ours[:-1, 0])

# Set the matplotlib settings
fig = plt.figure(figsize=(3.5, 6))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8
plt.rcParams["figure.autolayout"] = True
plt.rcParams['text.usetex'] = True
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[ees_blue, ees_yellow, ees_green, ees_red, ees_lightblue])


# Plot the results
plt.subplot(3, 1, 1)
plt.plot(data_ours[:, 0], data_ours[:, 1], color='C0',  linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 2] - 1, linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 5], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 6] - 1, linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 9], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 10] - 1, linestyle=':', color='C2', linewidth=2)
plt.ylabel('Speed $\Delta \omega$ (pu)')

plt.subplot(3, 1, 2)
plt.plot(data_ours[:, 0], data_ours[:, 2], color='C0', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 3], linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 6], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 7], linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 10], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 11], linestyle=':', color='C2', linewidth=2)
plt.ylabel('Electrical Power $P_e$ (pu)')

plt.subplot(3, 1, 3)
plt.plot(data_ours[:, 0], data_ours[:, 3], color='C0', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 4], linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 7], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 8], linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 11], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 12], linestyle=':', color='C2', linewidth=2)

plt.ylabel('Subtransient d-axis voltage (pu)')
plt.xlabel('Time (s)')

# Add the legend and labels to the last subplot
l1, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G1', color='C0', linewidth=1)
l2, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G2', color='C1', linewidth=1)
l3, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G3', color='C2', linewidth=1)
l5, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='Python', color='black', linewidth=1)
l6, = plt.plot(data_ours[0, 0], data_ours[0, 9], linestyle=':', label='PowerFactory', color='black', linewidth=2)

leg1 = plt.legend(bbox_to_anchor=(0., -0.38, 1.0, 0.1), handles=[l1, l2, l3], ncol=3, mode="expand", borderaxespad=0.)
ax = plt.gca().add_artist(leg1)
plt.legend(bbox_to_anchor=(0., -0.51, 1.0, 0.1), handles=[l5, l6], ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()
plt.savefig('./data/plots/comp_9bus.pdf')
plt.show()
