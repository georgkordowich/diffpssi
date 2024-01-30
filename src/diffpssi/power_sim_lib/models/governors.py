"""
File containing the TGOV1 model. Other governor models can be added here.
"""
from src.diffpssi.power_sim_lib.backend import *
from src.diffpssi.power_sim_lib.models.blocks import LeadLag, PT1Limited


class TGOV1(object):
    """
    Represents the TGOV1 (Turbine Governor) model in power system simulations.

    The TGOV1 model simulates the behavior of a turbine governor, which controls the mechanical power output
    of a turbine to maintain a stable system frequency. It includes a proportional controller (droop), a PT1
    limited block for governor action, and a lead-lag compensator for stability enhancement.

    Attributes:
        pt1_lim (PT1Limited): The PT1Limited block representing the governor action.
        lead_lag (LeadLag): The LeadLag block representing the compensator.
        p_ref (float or torch.Tensor): The reference power setpoint for the governor.
    """
    def __init__(self, param_dict, parallel_sims):
        """
        Initializes the TGOV1 model with specified parameters.

        Args:
            param_dict (dict, optional): A dictionary of parameters for the model.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """

        self.name = param_dict['name']
        self.r = param_dict['R']
        self.d_t = param_dict['D_t']
        self.t_1 = param_dict['T_1']
        self.t_2 = param_dict['T_2']
        self.t_3 = param_dict['T_3']
        self.v_min = param_dict['V_min']
        self.v_max = param_dict['V_max']

        droop = 1 / self.r
        self.pt1_lim = PT1Limited(t_pt1=self.t_1, gain_pt1=droop, lim_min=self.v_min, lim_max=self.v_max,
                                  parallel_sims=parallel_sims)
        self.lead_lag = LeadLag(t_1=self.t_2, t_2=self.t_3, parallel_sims=parallel_sims)

        self.p_ref = 0.0

    def differential(self):
        """
        Computes the differential equations for the TGOV1 model.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        return torch.concatenate([self.pt1_lim.differential(), self.lead_lag.differential()], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the TGOV1 model.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.concatenate([self.pt1_lim.get_state_vector(), self.lead_lag.get_state_vector()], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the TGOV1 model.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.pt1_lim.set_state_vector(x[:, 0:1])
        self.lead_lag.set_state_vector(x[:, 1:2])

    def get_output(self, omega_diff):
        """
        Computes the output of the TGOV1 model given the frequency deviation.

        Args:
            omega_diff (float or torch.Tensor): The deviation of the system frequency from its nominal value.

        Returns:
            torch.Tensor: The output of the model, representing the mechanical power command.
        """
        # "upper" path of the block diagram
        in1 = self.p_ref - omega_diff
        in2 = self.pt1_lim.get_output(in1)
        out1 = self.lead_lag.get_output(in2)

        # "damping path" of the block diagram
        out2 = self.d_t * omega_diff

        return out1 - out2

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by adjusting the reference power setpoint for the specified number of simulations.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.p_ref = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.p_ref

    def initialize(self, p_mech):
        """
        Initializes the TGOV1 model for simulation.

        Args:
            p_mech (float or torch.Tensor): The initial mechanical power.
        """
        # put the values here that shall come out of the blocks in the first time step so that all derivatives are zero
        in_1 = self.lead_lag.initialize(p_mech)
        self.p_ref = self.pt1_lim.initialize(in_1)
