"""
Unit tests for the simulation examples in the examples folder.
"""
import os
import unittest
import numpy as np


def set_env_vars():
    """
    Set the environment variables that are used to force the use of the heun integrator, the use of torch, and the
    number of parallel simulations.
    """
    # set environment variable that forces the simulator to always use the heun integrator
    os.environ['DIFFPSSI_FORCE_INTEGRATOR'] = 'heun'
    os.environ['DIFFPSSI_FORCE_SIM_BACKEND'] = 'torch'
    os.environ['DIFFPSSI_FORCE_OPT_ITERS'] = '1'
    os.environ['DIFFPSSI_FORCE_PARALLEL_SIMS'] = '2'


class TestOptRunAbility(unittest.TestCase):
    def test_custom_multi_machine_opt(self):
        set_env_vars()
        import examples.models.custom_multi_machine.custom_multi_machine_opt as custom_multi_machine_opt

        # get the path of the custom_multi_machine_sim.py file
        file_path = custom_multi_machine_opt.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the optimization
        custom_multi_machine_opt.main(parallel_sims=2)

    def test_ibb_opt(self):
        set_env_vars()
        import examples.models.ibb_model.ibb_opt as ibb_opt

        # get the path of the ibb_sim.py file
        file_path = ibb_opt.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the optimization
        ibb_opt.main(parallel_sims=2)

    def test_ieee_9bus_opt(self):
        set_env_vars()
        import examples.models.ieee_9bus.ieee_9bus_opt as ieee_9bus_opt

        # get the path of the ibb_sim.py file
        file_path = ieee_9bus_opt.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the optimization
        ieee_9bus_opt.main(parallel_sims=2)

    def test_k2a_opt(self):
        import examples.models.k2a.k2a_opt as k2a_opt

        # get the path of the ibb_sim.py file
        file_path = k2a_opt.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the optimization
        k2a_opt.main(parallel_sims=2)
