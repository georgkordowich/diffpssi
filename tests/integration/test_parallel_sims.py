"""
Unit tests for the simulation examples in the examples folder.
"""
import os
import unittest
import numpy as np

class TestSimulationOutput(unittest.TestCase):
    def set_env_vars(self):
        """
        Set the environment variables for the tests.
        """
        # set environment variable that forces the simulator to always use the heun integrator
        os.environ['DIFFPSSI_FORCE_INTEGRATOR'] = 'heun'
        os.environ['DIFFPSSI_TESTING'] = 'True'

        self.atol = 1e-8
        self.rtol = 1e-8

    def test_ieee_9bus_parallel_sim(self):
        self.set_env_vars()
        import examples.models.ieee_9bus.ieee_9bus_sim as ieee_9bus_sim

        # get the path of the ieee_9bus_sim.py file
        file_path = ieee_9bus_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        t, recorder = ieee_9bus_sim.main(parallel_sims=3)

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/ieee_9bus.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(recorder[0, :, :].real, expected_output, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(recorder[1, :, :].real, expected_output, atol=self.atol, rtol=self.rtol))
        self.assertTrue(np.allclose(recorder[2, :, :].real, expected_output, atol=self.atol, rtol=self.rtol))
