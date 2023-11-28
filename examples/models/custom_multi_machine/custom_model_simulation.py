import time

import numpy as np
import matplotlib.pyplot as plt
import examples.models.custom_multi_machine.custom_multi_machine as mdl

parallel_sims = 10

sim = mdl.get_model(parallel_sims)

sim.add_sc_event(1, 1.1, 'Bus 1')


def record_desired_parameters(simulation):
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

start_time = time.time()

t, recorder = sim.run()

end_time = time.time()
print('Dynamic simulation finished in {:.2f} seconds'.format(end_time - start_time))

# Format shall be [batch, timestep, value
plt.plot(t, recorder[0, :, :])
plt.legend([
    'IBB',
    'Gen1',
    'Gen2',
    'Gen3',
    'Gen4',
])
plt.show()

np.save('data/original_data.npy', recorder[0].real)
