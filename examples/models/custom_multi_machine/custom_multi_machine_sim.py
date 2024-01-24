"""
This example shows how to simulate a custom multi machine model.
"""
import numpy as np
import matplotlib.pyplot as plt
import examples.models.custom_multi_machine.custom_multi_machine_model as mdl
from power_sim_lib.simulator import PowerSystemSimulation as Pss

parallel_sims = 1

sim = Pss(parallel_sims=parallel_sims,
          sim_time=10,
          time_step=0.005,
          solver='euler',
          grid_data=mdl.load(),
          )

sim.add_sc_event(1, 1.1, 'Bus 1')


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

        simulation.busses[2].models[0].omega.real,
        simulation.busses[2].models[0].e_q_st.real,
        simulation.busses[2].models[0].e_d_st.real,

        simulation.busses[3].models[0].omega.real,
        simulation.busses[3].models[0].e_q_st.real,
        simulation.busses[3].models[0].e_d_st.real,

        simulation.busses[4].models[0].omega.real,
        simulation.busses[4].models[0].e_q_st.real,
        simulation.busses[4].models[0].e_d_st.real,
    ]
    return record_list


sim.set_record_function(record_desired_parameters)
t, recorder = sim.run()

# Format shall be [batch, timestep, value]
plt.plot(t, recorder[0, :, ::3])
plt.legend([
    'Gen1',
    'Gen2',
    'Gen3',
    'Gen4',
])
plt.show()

np.save('data/original_data.npy', recorder[0].real)
