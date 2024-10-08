"""
Contains the SEXS (Simplified Excitation System) model for power system simulations.
Other models can be added here as well.
"""
from src.diffpssi.power_sim_lib.backend import *
from src.diffpssi.power_sim_lib.models.blocks import LeadLag, PT1Limited


class SEXS(object):
    """
    Represents the SEXS (Simplified Excitation System) model in power system simulations.

    This class models a simplified excitation system for synchronous generators, combining lead-lag and
    first-order transfer function with limited output. It's used to simulate the dynamic behavior of the
    generator's voltage control system.

    Attributes:
        lead_lag (LeadLag): The lead-lag control block.
        pt1_lim (PT1Limited): The PT1 transfer function block with output limiting.
        v_setpoint (float or torch.Tensor): The setpoint for the system voltage.
        bias (float or torch.Tensor): A bias value for voltage control.
    """

    def __init__(self, param_dict=None,
                 name=None,
                 gen=None,
                 t_a=None,
                 t_b=None,
                 t_e=None,
                 k=None,
                 e_min=None,
                 e_max=None,
                 ):
        """
        Initializes the SEXS model with specified parameters.

        Args:
            param_dict (dict, optional): A dictionary of parameters for the model.
            name (str, optional): The name of the model.
            gen (str, optional): The name of the generator the model is connected to.
            t_a (float, optional): The time constant t1 of the lead-lag control block.
            t_b (float, optional): The time constant t2 of the lead-lag control block.
            t_e (float, optional): The time constant t of the PT1 transfer function block.
            k (float, optional): The gain of the PT1 transfer function block.
            e_min (float, optional): The minimum value of the output of the PT1 transfer function block.
            e_max (float, optional): The maximum value of the output of the PT1 transfer function block.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'gen': gen,
                'T_a': t_a,
                'T_b': t_b,
                'T_e': t_e,
                'K': k,
                'E_min': e_min,
                'E_max': e_max,
            }
        self.name = param_dict['name']
        self.gen = param_dict['gen']
        self.t_a = param_dict['T_a']
        self.t_b = param_dict['T_b']
        self.t_e = param_dict['T_e']
        self.k = param_dict['K']
        self.e_min = param_dict['E_min']
        self.e_max = param_dict['E_max']

        self.lead_lag = LeadLag(t_1=self.t_a, t_2=self.t_b)
        self.pt1_lim = PT1Limited(t_pt1=self.t_e, gain_pt1=self.k, lim_min=self.e_min, lim_max=self.e_max)

        self.v_setpoint = 0.0
        self.bias = 0.0

    def differential(self):
        """
        Computes the differential equations for the SEXS model.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        return torch.concatenate([self.lead_lag.differential(), self.pt1_lim.differential()], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the SEXS model.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.concatenate([self.lead_lag.get_state_vector(), self.pt1_lim.get_state_vector()], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the SEXS model.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.lead_lag.set_state_vector(x[:, 0:1])
        self.pt1_lim.set_state_vector(x[:, 1:2])

    def get_output(self, input_value):
        """
        Computes the output of the SEXS model given an input value.

        Args:
            input_value (float or torch.Tensor): The input value to the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        in1 = self.v_setpoint - input_value + self.bias
        in2 = self.lead_lag.get_output(in1)
        output = self.pt1_lim.get_output(in2)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.v_setpoint = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_setpoint
        self.bias = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.bias
        self.lead_lag.enable_parallel_simulation(parallel_sims)
        self.pt1_lim.enable_parallel_simulation(parallel_sims)

    def initialize(self, e_fd):
        """
        Initializes the SEXS model for simulation.

        Args:
            e_fd (float or torch.Tensor): The initial value for the field voltage.
        """
        # put the values here that shall come out of the blocks in the first time step so that all derivatives are zero
        in1 = self.pt1_lim.initialize(e_fd)
        self.bias = self.lead_lag.initialize(in1)
