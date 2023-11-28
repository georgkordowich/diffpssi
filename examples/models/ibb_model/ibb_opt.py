import numpy as np
import torch
import examples.models.ibb_model.ibb_model as mdl
from optimization_lib.ps_optimization import PowerSystemOptimization

np.random.seed(0)

parallel_sims = 100

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

sim.busses[1].models[0].h = torch.tensor(np.random.uniform(0.5 * 3.5, 2.0 * 3.5, (parallel_sims, 1)), requires_grad=True, dtype=torch.complex128)
sim.busses[1].models[0].x_d = torch.tensor(np.random.uniform(0.5 * 1.81, 2.0 * 1.81, (parallel_sims, 1)), requires_grad=True, dtype=torch.complex128)
sim.busses[1].models[0].x_q = torch.tensor(np.random.uniform(0.5 * 1.76, 2.0 * 1.76, (parallel_sims, 1)), requires_grad=True, dtype=torch.complex128)
sim.busses[1].models[0].x_d_t = torch.tensor(np.random.uniform(0.5 * 0.3, 2.0 * 0.3, (parallel_sims, 1)), requires_grad=True, dtype=torch.complex128)
sim.busses[1].models[0].x_q_t = torch.tensor(np.random.uniform(0.5 * 0.65, 2.0 * 0.65, (parallel_sims, 1)), requires_grad=True, dtype=torch.complex128)

optimizable_parameters = [
    sim.busses[1].models[0].h,
    sim.busses[1].models[0].x_d,
    sim.busses[1].models[0].x_q,
    sim.busses[1].models[0].x_d_t,
    sim.busses[1].models[0].x_q_t,
]

param_names = [
    'Gen1 h',
    'Gen1 x_d',
    'Gen1 x_q',
    'Gen1 x_d_t',
    'Gen1 x_q_t',
]

params_original = [
    3.5,
    1.81,
    1.76,
    0.3,
    0.65,
]

original_data = np.load('data/original_data.npy')
orig_tensor = torch.tensor(original_data) * torch.ones((parallel_sims,) + original_data.shape)

# mute sim because it does not generate added value
sim.verbose = False

opt = PowerSystemOptimization(sim,
                              orig_tensor,
                              params_optimizable=optimizable_parameters,
                              params_original=params_original,
                              param_names=param_names,
                              decay=0.75,
                              )

opt.run()