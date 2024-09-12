"""
This example shows how to simulate the IEEE 9 bus system.
"""
import numpy as np
import matplotlib.pyplot as plt
import examples.models.ieee_9bus.ieee_9bus_model as mdl
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    # Record the desired parameters
    record_list = [
        simulation.busses[0].models[0].omega.real,
        simulation.busses[0].models[0].p_e,
        simulation.busses[0].models[0].v_bb.real,
        simulation.busses[0].models[0].v_bb.imag,

        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].p_e,
        simulation.busses[1].voltage.real,
        simulation.busses[1].voltage.imag,

        simulation.busses[2].models[0].omega.real,
        simulation.busses[2].models[0].p_e,
        simulation.busses[2].voltage.real,
        simulation.busses[2].voltage.imag,
    ]
    return record_list


def main(parallel_sims=1):
    """
    This function simulates the IEEE 9 bus system.
    """
    sim = Pss(parallel_sims=parallel_sims,
              sim_time=5,
              time_step=0.005,
              solver='heun',
              grid_data=mdl.load(),
              )

    sim.add_sc_event(1, 1.05, 'B8')

    sim.set_record_function(record_desired_parameters)
    t, recorder = sim.run()

    # Format shall be [batch, timestep, value]
    # create a new subplot for each parameter
    plt.figure()

    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel('Parameter {}'.format(i))
        plt.xlabel('Time [s]')

    plt.show()

    # add time to the first column
    saver = np.zeros((len(t), len(recorder[0, 0, :]) + 1))
    saver[:, 0] = t
    saver[:, 1:] = recorder[0, :, :].real

    np.save('data/original_data.npy', recorder[0].real)
    np.save('data/original_data_t.npy', saver)

    return t, recorder


if __name__ == '__main__':
    main()
