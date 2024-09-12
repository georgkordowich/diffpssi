"""
Unit tests for the simulation examples in the examples folder.
"""
import unittest


class TestCorrectBackend(unittest.TestCase):
    def test_correct_backend(self):
        """
        This test checks if the correct backend is used.
        """
        import src.diffpssi.power_sim_lib.backend as backend

        self.assertEqual(backend.BACKEND, 'numpy')
