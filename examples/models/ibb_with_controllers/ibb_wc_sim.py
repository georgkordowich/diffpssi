"""
This example shows how to simulate the IBB model with controllers.
"""
import numpy as np
import matplotlib.pyplot as plt
import examples.models.ibb_with_controllers.ibb_wc_model as mdl
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss
from src.diffpssi.power_sim_lib.backend import *


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    # Record the desired parameters
    record_list = [
        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_d_st.real,
    ]
    return record_list


def main():
    """
    This function simulates the IBB with controllers model.
    """
    parallel_sims = 1

    sim = Pss(parallel_sims=parallel_sims,
              sim_time=10,
              time_step=0.005,
              solver='heun',
              grid_data=mdl.load(),
              )

    sim.add_sc_event(1, 1.05, 'B2')

    sim.set_record_function(record_desired_parameters)
    t, recorder = sim.run()

    # Format shall be [batch, timestep, value]
    plt.figure()

    recorder = torch.unsqueeze(recorder[:, :, 0], -1)

    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel('Parameter {}'.format(i))
        plt.xlabel('Time [s]')

    plt.show()

    np.save('data/original_data.npy', recorder[0].real)


if __name__ == '__main__':
    main()
