"""
Unit tests for the simulation examples in the examples folder.
"""
import os
import unittest
import numpy as np

# set environment variable that forces the simulator to always use the heun integrator
os.environ['DIFFPSSI_FORCE_INTEGRATOR'] = 'heun'

atol = 1e-8
rtol = 1e-8


class TestCustomMultiMachine(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.custom_multi_machine.custom_multi_machine_sim as custom_multi_machine_sim

        # get the path of the custom_multi_machine_sim.py file
        file_path = custom_multi_machine_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        custom_multi_machine_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = custom_multi_machine_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/custom_multi_machine.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))


class TestIbbModel(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.ibb_model.ibb_sim as ibb_sim

        # get the path of the ibb_sim.py file
        file_path = ibb_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        ibb_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = ibb_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/ibb_model.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))


class TestIbbTransformer(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.ibb_transformer.ibb_trans_sim as ibb_trans_sim

        # get the path of the ibb_trans_sim.py file
        file_path = ibb_trans_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        ibb_trans_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = ibb_trans_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/ibb_transformer.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))


class TestIbbWithControllers(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.ibb_with_controllers.ibb_wc_sim as ibb_wc_sim

        # get the path of the ibb_wc_sim.py file
        file_path = ibb_wc_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        ibb_wc_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = ibb_wc_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/ibb_with_controllers.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))


class TestIEEE9Bus(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.ieee_9bus.ieee_9bus_sim as ieee_9bus_sim

        # get the path of the ieee_9bus_sim.py file
        file_path = ieee_9bus_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        ieee_9bus_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = ieee_9bus_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/ieee_9bus.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))


class TestK2A(unittest.TestCase):
    def test_simulation_output(self):
        import examples.models.k2a.k2a_sim as k2a_sim

        # get the path of the k2a_sim.py file
        file_path = k2a_sim.__file__

        os.chdir(os.path.dirname(file_path))

        # Run the simulation
        k2a_sim.main()

        # Load the output file and compare it to the expected output
        output_sim = k2a_sim.np.load('./data/original_data.npy')

        # change the directory back to where this file here is
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load the expected output in the tests folder
        expected_output = np.load('./data/k2a.npy')

        # compare the two outputs
        self.assertTrue(np.allclose(output_sim, expected_output, atol=atol, rtol=rtol))
