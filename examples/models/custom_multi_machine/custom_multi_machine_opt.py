"""
This example shows how to optimize the parameters of a custom multi-machine model.
"""
import numpy as np
import torch

import examples.models.custom_multi_machine.custom_multi_machine_model as mdl
from src.diffpssi.optimization_lib.ps_optimization import PowerSystemOptimization
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation

np.random.seed(0)


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


def main(parallel_sims=1000):
    """
    This function runs a parameter identification for the custom multi-machine model.
    """
    parallel_sims = parallel_sims

    sim = PowerSystemSimulation(parallel_sims=parallel_sims,
                                sim_time=10,
                                time_step=0.005,
                                solver='euler',
                                grid_data=mdl.load(),
                                )

    sim.add_sc_event(1, 1.1, 'Bus 1')

    sim.set_record_function(record_desired_parameters)

    sim.busses[1].models[0].h = torch.tensor(np.random.uniform(0.5 * 6, 2.0 * 6, (parallel_sims, 1)),
                                             requires_grad=True,
                                             dtype=torch.complex128)
    sim.busses[1].models[0].x_d = torch.tensor(np.random.uniform(0.5 * 1.81, 2.0 * 1.81, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[1].models[0].x_q = torch.tensor(np.random.uniform(0.5 * 1.76, 2.0 * 1.76, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[1].models[0].x_d_t = torch.tensor(np.random.uniform(0.5 * 0.3, 2.0 * 0.3, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)
    sim.busses[1].models[0].x_q_t = torch.tensor(np.random.uniform(0.5 * 0.65, 2.0 * 0.65, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)

    sim.busses[2].models[0].h = torch.tensor(np.random.uniform(0.5 * 4.9, 2.0 * 4.9, (parallel_sims, 1)),
                                             requires_grad=True, dtype=torch.complex128)
    sim.busses[2].models[0].x_d = torch.tensor(np.random.uniform(0.5 * 1.81, 2.0 * 1.81, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[2].models[0].x_q = torch.tensor(np.random.uniform(0.5 * 1.76, 2.0 * 1.76, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[2].models[0].x_d_t = torch.tensor(np.random.uniform(0.5 * 0.3, 2.0 * 0.3, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)
    sim.busses[2].models[0].x_q_t = torch.tensor(np.random.uniform(0.5 * 0.65, 2.0 * 0.65, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)

    sim.busses[3].models[0].h = torch.tensor(np.random.uniform(0.5 * 4.1, 2.0 * 4.9, (parallel_sims, 1)),
                                             requires_grad=True, dtype=torch.complex128)
    sim.busses[3].models[0].x_d = torch.tensor(np.random.uniform(0.5 * 1.81, 2.0 * 1.81, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[3].models[0].x_q = torch.tensor(np.random.uniform(0.5 * 1.76, 2.0 * 1.76, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[3].models[0].x_d_t = torch.tensor(np.random.uniform(0.5 * 0.3, 2.0 * 0.3, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)
    sim.busses[3].models[0].x_q_t = torch.tensor(np.random.uniform(0.5 * 0.65, 2.0 * 0.65, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)

    sim.busses[4].models[0].h = torch.tensor(np.random.uniform(0.5 * 3.2, 2.0 * 4.9, (parallel_sims, 1)),
                                             requires_grad=True, dtype=torch.complex128)
    sim.busses[4].models[0].x_d = torch.tensor(np.random.uniform(0.5 * 1.81, 2.0 * 1.81, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[4].models[0].x_q = torch.tensor(np.random.uniform(0.5 * 1.76, 2.0 * 1.76, (parallel_sims, 1)),
                                               requires_grad=True, dtype=torch.complex128)
    sim.busses[4].models[0].x_d_t = torch.tensor(np.random.uniform(0.5 * 0.3, 2.0 * 0.3, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)
    sim.busses[4].models[0].x_q_t = torch.tensor(np.random.uniform(0.5 * 0.65, 2.0 * 0.65, (parallel_sims, 1)),
                                                 requires_grad=True, dtype=torch.complex128)

    optimizable_parameters = [
        sim.busses[1].models[0].h,
        sim.busses[1].models[0].x_d,
        sim.busses[1].models[0].x_q,
        sim.busses[1].models[0].x_d_t,
        sim.busses[1].models[0].x_q_t,

        sim.busses[2].models[0].h,
        sim.busses[2].models[0].x_d,
        sim.busses[2].models[0].x_q,
        sim.busses[2].models[0].x_d_t,
        sim.busses[2].models[0].x_q_t,

        sim.busses[3].models[0].h,
        sim.busses[3].models[0].x_d,
        sim.busses[3].models[0].x_q,
        sim.busses[3].models[0].x_d_t,
        sim.busses[3].models[0].x_q_t,

        sim.busses[4].models[0].h,
        sim.busses[4].models[0].x_d,
        sim.busses[4].models[0].x_q,
        sim.busses[4].models[0].x_d_t,
        sim.busses[4].models[0].x_q_t,
    ]

    param_names = [
        'Gen1 h',
        'Gen1 x_d',
        'Gen1 x_q',
        'Gen1 x_d_t',
        'Gen1 x_q_t',

        'Gen2 h',
        'Gen2 x_d',
        'Gen2 x_q',
        'Gen2 x_d_t',
        'Gen2 x_q_t',

        'Gen3 h',
        'Gen3 x_d',
        'Gen3 x_q',
        'Gen3 x_d_t',
        'Gen3 x_q_t',

        'Gen4 h',
        'Gen4 x_d',
        'Gen4 x_q',
        'Gen4 x_d_t',
        'Gen4 x_q_t',
    ]

    params_original = [
        6.0,
        1.81,
        1.76,
        0.3,
        0.65,

        4.9,
        1.81,
        1.76,
        0.3,
        0.65,

        4.1,
        1.81,
        1.76,
        0.3,
        0.65,

        3.2,
        1.81,
        1.76,
        0.3,
        0.65,
    ]

    # mute sim because it does not generate added value
    sim.verbose = False

    original_data = np.load('data/original_data.npy')
    orig_tensor = torch.tensor(original_data) * torch.ones((parallel_sims,) + original_data.shape)

    opt = PowerSystemOptimization(sim,
                                  orig_tensor,
                                  params_optimizable=optimizable_parameters,
                                  params_original=params_original,
                                  param_names=param_names,
                                  )

    opt.run()


if __name__ == '__main__':
    main()
