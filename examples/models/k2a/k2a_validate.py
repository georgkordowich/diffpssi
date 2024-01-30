"""
This script compares the results of the Python model with the results of the PowerFactory model.
"""
import math

import numpy as np
import matplotlib.pyplot as plt

path_data_mine = r'data/original_data_t.npy'
path_power_factory = r'data/pf_data.txt'
data_ours = np.load(path_data_mine)
data_pf = np.loadtxt(path_power_factory, skiprows=2)
length = len(data_ours[:-1, 0])

fig = plt.figure(figsize=(3.5, 6))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8
plt.rcParams["figure.autolayout"] = True
plt.rcParams['text.usetex'] = True

# Plot the first subplot
plt.subplot(3, 1, 1)
plt.plot(data_ours[:, 0], data_ours[:, 1], color='C0', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 1] + math.pi/2, linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 2], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 2] + math.pi/2, linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 3], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 3] + math.pi/2, linestyle=':', color='C2', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 4], color='C3', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 4] + math.pi/2, linestyle=':', color='C3', linewidth=2)
plt.ylabel('Angle $\delta$ (rad)')


plt.subplot(3, 1, 2)
plt.plot(data_ours[:, 0], data_ours[:, 5], color='C0', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 5] - 1, linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 6], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 6] - 1, linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 7], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 7] - 1, linestyle=':', color='C2', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 8], color='C3', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 8] - 1, linestyle=':', color='C3', linewidth=2)
plt.ylabel('Speed $\Delta \omega$ (pu)')


plt.subplot(3, 1, 3)
plt.plot(data_ours[:, 0], data_ours[:, 9], color='C0', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 9], linestyle=':', color='C0', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 10], color='C1', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 10], linestyle=':', color='C1', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 11], color='C2', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 11], linestyle=':', color='C2', linewidth=2)

plt.plot(data_ours[:, 0], data_ours[:, 12], color='C3', linewidth=1)
plt.plot(data_pf[:length, 0], data_pf[:length, 12], linestyle=':', color='C3', linewidth=2)
plt.ylabel('Subtransient q-axis voltage (pu)')
plt.xlabel('Time (s)')

# Add the legend and labels to the last subplot
l1, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G1', color='C0', linewidth=1)
l2, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G2', color='C1', linewidth=1)
l3, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G3', color='C2', linewidth=1)
l4, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='G4', color='C3', linewidth=1)
l5, = plt.plot(data_ours[0, 0], data_ours[0, 9], label='Python', color='black', linewidth=1)
l6, = plt.plot(data_ours[0, 0], data_ours[0, 9], linestyle=':', label='PowerFactory', color='black', linewidth=2)

leg1 = plt.legend(bbox_to_anchor=(0., -0.38, 1.0, 0.1),
                  handles=[l1, l2, l3, l4],
                  ncol=4,
                  mode="expand",
                  borderaxespad=0.)
ax = plt.gca().add_artist(leg1)
plt.legend(bbox_to_anchor=(0., -0.51, 1.0, 0.1), handles=[l5, l6], ncol=2, mode="expand", borderaxespad=0.)

plt.tight_layout()
plt.savefig('./data/plots/comp_k2a.pdf')
plt.show()
