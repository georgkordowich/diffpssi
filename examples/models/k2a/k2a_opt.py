"""
File contains an example of how to use the PowerSystemOptimization class to optimize the parameters of a power system
stabilizer model with respect to system stability.
"""
import numpy as np
import torch

import examples.models.k2a.k2a_model as mdl
from diffpssi.optimization_lib.ps_optimization import PowerSystemOptimization
from diffpssi.power_sim_lib.simulator import PowerSystemSimulation

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
        simulation.busses[0].models[0].omega.real,
        simulation.busses[1].models[0].omega.real,
        simulation.busses[2].models[0].omega.real,
        simulation.busses[3].models[0].omega.real,
    ]
    return record_list


def main(parallel_sims=10):
    """
    This function runs a parameter optimization of PSS parameters for the K2A model.
    """
    parallel_sims = parallel_sims

    sim = PowerSystemSimulation(parallel_sims=parallel_sims,
                                sim_time=10,
                                time_step=0.005,
                                solver='euler',
                                grid_data=mdl.load(),
                                )

    sim.add_sc_event(1, 1.1, 'B1')

    sim.set_record_function(record_desired_parameters)

    param_names = [
        'G1 stab1 kw',
        'G1 stab1 tw',
        'G1 stab1 t1',
        'G1 stab1 t2',
        'G1 stab1 t3',
        'G1 stab1 t4',

        'G2 stab1 kw',
        'G2 stab1 tw',
        'G2 stab1 t1',
        'G2 stab1 t2',
        'G2 stab1 t3',
        'G2 stab1 t4',

        'G3 stab1 kw',
        'G3 stab1 tw',
        'G3 stab1 t1',
        'G3 stab1 t2',
        'G3 stab1 t3',
        'G3 stab1 t4',

        'G4 stab1 kw',
        'G4 stab1 tw',
        'G4 stab1 t1',
        'G4 stab1 t2',
        'G4 stab1 t3',
        'G4 stab1 t4',
    ]

    params_original = [
        50.0,
        10.0,
        0.5,
        0.5,
        0.05,
        0.05,

        50.0,
        10.0,
        0.5,
        0.5,
        0.05,
        0.05,

        50.0,
        10.0,
        0.5,
        0.5,
        0.05,
        0.05,

        50.0,
        10.0,
        0.5,
        0.5,
        0.05,
        0.05,
    ]

    sim.busses[0].models[0].stabilizer.washout.k_w = torch.tensor(
        np.random.uniform(0.5 * params_original[0], 2.0 * params_original[0], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[0].models[0].stabilizer.washout.t_w = torch.tensor(
        np.random.uniform(0.5 * params_original[1], 2.0 * params_original[1], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[0].models[0].stabilizer.lead_lag1.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[2], 2.0 * params_original[2], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[0].models[0].stabilizer.lead_lag1.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[4], 2.0 * params_original[4], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[0].models[0].stabilizer.lead_lag2.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[3], 2.0 * params_original[3], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[0].models[0].stabilizer.lead_lag2.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[5], 2.0 * params_original[5], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)

    sim.busses[1].models[0].stabilizer.washout.k_w = torch.tensor(
        np.random.uniform(0.5 * params_original[6], 2.0 * params_original[6], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[1].models[0].stabilizer.washout.t_w = torch.tensor(
        np.random.uniform(0.5 * params_original[7], 2.0 * params_original[7], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[1].models[0].stabilizer.lead_lag1.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[8], 2.0 * params_original[8], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[1].models[0].stabilizer.lead_lag1.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[10], 2.0 * params_original[10], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[1].models[0].stabilizer.lead_lag2.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[9], 2.0 * params_original[9], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[1].models[0].stabilizer.lead_lag2.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[11], 2.0 * params_original[11], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)

    sim.busses[2].models[0].stabilizer.washout.k_w = torch.tensor(
        np.random.uniform(0.5 * params_original[12], 2.0 * params_original[12], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[2].models[0].stabilizer.washout.t_w = torch.tensor(
        np.random.uniform(0.5 * params_original[13], 2.0 * params_original[13], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[2].models[0].stabilizer.lead_lag1.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[14], 2.0 * params_original[14], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[2].models[0].stabilizer.lead_lag1.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[16], 2.0 * params_original[16], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[2].models[0].stabilizer.lead_lag2.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[15], 2.0 * params_original[15], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[2].models[0].stabilizer.lead_lag2.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[17], 2.0 * params_original[17], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)

    sim.busses[3].models[0].stabilizer.washout.k_w = torch.tensor(
        np.random.uniform(0.5 * params_original[18], 2.0 * params_original[18], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[3].models[0].stabilizer.washout.t_w = torch.tensor(
        np.random.uniform(0.5 * params_original[19], 2.0 * params_original[19], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[3].models[0].stabilizer.lead_lag1.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[20], 2.0 * params_original[20], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[3].models[0].stabilizer.lead_lag1.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[22], 2.0 * params_original[22], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[3].models[0].stabilizer.lead_lag2.t_1 = torch.tensor(
        np.random.uniform(0.5 * params_original[21], 2.0 * params_original[21], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)
    sim.busses[3].models[0].stabilizer.lead_lag2.t_2 = torch.tensor(
        np.random.uniform(0.5 * params_original[23], 2.0 * params_original[23], (parallel_sims, 1)), requires_grad=True,
        dtype=torch.complex128)

    optimizable_parameters = [
        sim.busses[0].models[0].stabilizer.washout.k_w,
        sim.busses[0].models[0].stabilizer.washout.t_w,
        sim.busses[0].models[0].stabilizer.lead_lag1.t_1,
        sim.busses[0].models[0].stabilizer.lead_lag1.t_2,
        sim.busses[0].models[0].stabilizer.lead_lag2.t_1,
        sim.busses[0].models[0].stabilizer.lead_lag2.t_2,

        sim.busses[1].models[0].stabilizer.washout.k_w,
        sim.busses[1].models[0].stabilizer.washout.t_w,
        sim.busses[1].models[0].stabilizer.lead_lag1.t_1,
        sim.busses[1].models[0].stabilizer.lead_lag1.t_2,
        sim.busses[1].models[0].stabilizer.lead_lag2.t_1,
        sim.busses[1].models[0].stabilizer.lead_lag2.t_2,

        sim.busses[2].models[0].stabilizer.washout.k_w,
        sim.busses[2].models[0].stabilizer.washout.t_w,
        sim.busses[2].models[0].stabilizer.lead_lag1.t_1,
        sim.busses[2].models[0].stabilizer.lead_lag1.t_2,
        sim.busses[2].models[0].stabilizer.lead_lag2.t_1,
        sim.busses[2].models[0].stabilizer.lead_lag2.t_2,

        sim.busses[3].models[0].stabilizer.washout.k_w,
        sim.busses[3].models[0].stabilizer.washout.t_w,
        sim.busses[3].models[0].stabilizer.lead_lag1.t_1,
        sim.busses[3].models[0].stabilizer.lead_lag1.t_2,
        sim.busses[3].models[0].stabilizer.lead_lag2.t_1,
        sim.busses[3].models[0].stabilizer.lead_lag2.t_2,
    ]

    # mute sim because it does not generate added value
    sim.verbose = False

    # batch, timestep, value
    target_tensor = torch.zeros((parallel_sims, int(10 / 0.005), len(record_desired_parameters(sim))))

    # only consider the stuff after the fault is cleared
    def loss_function(sim_result, target_data):
        """
        Calculates the loss function for the optimization.
        Args:
            sim_result: The result of the simulation.
            target_data: The target data.

        Returns: The value of the loss function.

        """
        # noinspection PyArgumentList
        return torch.mean(
            torch.sum(torch.abs(target_data[:, int(1.1 / 0.005):, :] - sim_result[:, int(1.1 / 0.005):, :]), dim=2),
            axis=1)

    opt = PowerSystemOptimization(sim,
                                  target_tensor,
                                  params_optimizable=optimizable_parameters,
                                  param_names=param_names,
                                  enable_plots=True,
                                  loss_function=loss_function,
                                  )

    opt.run()


if __name__ == '__main__':
    main()
