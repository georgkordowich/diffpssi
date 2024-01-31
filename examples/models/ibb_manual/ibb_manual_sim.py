"""
This example shows how to manually create and simulate the IBB model.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

from diffpssi.power_sim_lib.models.synchronous_machine import SynchMachine
from diffpssi.power_sim_lib.models.static_models import *


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
    This function simulates the IBB model.
    """
    parallel_sims = 1

    sim = Pss(parallel_sims=parallel_sims,
              sim_time=10,
              time_step=0.005,
              solver='heun',
              )

    sim.fn = 60
    sim.base_mva = 2200
    sim.base_voltage = 24

    sim.add_bus(Bus(name='Bus 0', v_n=24))
    sim.add_bus(Bus(name='Bus 1', v_n=24))

    sim.add_line(Line(name='L1', from_bus='Bus 0', to_bus='Bus 1', length=1, s_n=2200, v_n=24, unit='p.u.',
                      r=0, x=0.65, b=0, s_n_sys=2200, v_n_sys=24))

    sim.add_generator(SynchMachine(name='IBB', bus='Bus 0', s_n=22000, v_n=24, p=-1998, v=0.995, h=3.5e7, d=0,
                                   x_d=1.81, x_q=1.76, x_d_t=0.3, x_q_t=0.65, x_d_st=0.23, x_q_st=0.23, t_d0_t=8.0,
                                   t_q0_t=1, t_d0_st=0.03, t_q0_st=0.07, f_n_sys=60, s_n_sys=2200, v_n_sys=24))
    sim.add_generator(SynchMachine(name='Gen 1', bus='Bus 1', s_n=2200, v_n=24, p=1998, v=1, h=3.5, d=0, x_d=1.81,
                                   x_q=1.76, x_d_t=0.3, x_q_t=0.65, x_d_st=0.23, x_q_st=0.23, t_d0_t=8.0, t_q0_t=1,
                                   t_d0_st=0.03, t_q0_st=0.07, f_n_sys=60, s_n_sys=2200, v_n_sys=24))

    sim.set_slack_bus('Bus 0')

    sim.add_sc_event(1, 1.05, 'Bus 1')
    sim.set_record_function(record_desired_parameters)

    # Run the simulation. Recorder format shall be [batch, timestep, value]
    t, recorder = sim.run()

    # Plot the results
    plt.figure()
    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel('Parameter {}'.format(i))
        plt.xlabel('Time [s]')
    plt.show()

    if os.environ.get('DIFFPSSI_TESTING') == 'True':
        np.save('./data/original_data.npy', recorder[0].real)


if __name__ == '__main__':
    main()
