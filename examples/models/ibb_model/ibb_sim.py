import time

import numpy as np
import matplotlib.pyplot as plt
import examples.models.ibb_model.ibb_model as mdl

parallel_sims = 10

sim = mdl.get_model(parallel_sims)

sim.add_sc_event(1, 1.05, 'Bus 1')


def record_desired_parameters(simulation):
    # Record the desired parameters
    record_list = [
        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_d_st.real,
    ]
    return record_list


sim.set_record_function(record_desired_parameters)

start_time = time.time()

t, recorder = sim.run()

end_time = time.time()
print('Dynamic simulation finished in {:.2f} seconds'.format(end_time - start_time))

# Format shall be [batch, timestep, value]
# create a new subplot for each parameter
plt.figure()

for i in range(len(recorder[0, 0, :])):
    plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
    plt.plot(t, recorder[0, :, i].real)
    plt.ylabel('Parameter {}'.format(i))
    plt.xlabel('Time [s]')

plt.show()

np.save('data/original_data.npy', recorder[0].real)
