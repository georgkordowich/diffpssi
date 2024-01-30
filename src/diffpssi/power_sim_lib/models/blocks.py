"""
File contains implemented controller blocks for power system simulations.
"""
from src.diffpssi.power_sim_lib.backend import *


class PT1Limited(object):
    """
    Represents a first-order transfer function with output limiting (PT1Limited) in power system simulations.

    This class models a first-order lag system with a linear gain and limits on the output. It's useful in
    situations where a system's response needs to be limited within a specific range.

    Attributes:
        t_pt1 (float or torch.Tensor): Time constant of the PT1 transfer function.
        gain_pt1 (float or torch.Tensor): Gain of the PT1 transfer function.
        lim_min (float or torch.Tensor): Minimum limit for the output.
        lim_max (float or torch.Tensor): Maximum limit for the output.
        input (float or torch.Tensor): Current input to the PT1Limited block.
        state_1 (float or torch.Tensor): Internal state of the PT1Limited block.
    """

    def __init__(self, t_pt1, gain_pt1, lim_min, lim_max, parallel_sims=None):
        """
        Initializes the PT1Limited block with specified parameters.

        Args:
            t_pt1 (float): Time constant of the PT1 transfer function.
            gain_pt1 (float): Gain of the PT1 transfer function.
            lim_min (float): Minimum limit for the output.
            lim_max (float): Maximum limit for the output.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.t_pt1 = t_pt1
        self.gain_pt1 = gain_pt1
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.enable_parallel_simulation(parallel_sims)

        self.input = 0

        self.state_1 = 0

    def differential(self):
        """
        Computes the differential equations for the PT1Limited model.
        Returns: A tensor containing the derivatives of the state variables.
        """
        dx1 = 1 / self.t_pt1 * (self.gain_pt1 * self.input - self.state_1)
        # noinspection PyTypeChecker
        dx1 = torch.where(torch.logical_or(torch.logical_and(self.state_1.real <= self.lim_min,
                                                             dx1.real < 0),
                                           torch.logical_and(self.state_1.real >= self.lim_max,
                                                             dx1.real > 0)),
                          0, dx1)
        return torch.stack([dx1, ], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the PT1Limited model.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.stack([self.state_1, ], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the PT1Limited model.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Computes the output of the PT1Limited model.
        Args:
            input_var (torch.Tensor): The current input to the model.

        Returns: The output of the model.

        """
        self.input = input_var
        # noinspection PyTypeChecker
        output = torch.minimum(torch.maximum(self.state_1.real, self.lim_min), self.lim_max)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.t_pt1 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_pt1
        self.gain_pt1 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.gain_pt1
        self.lim_min = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_min
        self.lim_max = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.lim_max

    def initialize(self, out_wish):
        """
        Initializes the PT1Limited model with a specified output.
        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.

        """
        # noinspection PyTypeChecker
        self.state_1 = torch.minimum(torch.maximum(out_wish.real, self.lim_min), self.lim_max)
        return out_wish / self.gain_pt1


class Limiter(object):
    """
    Represents a simple limiter block in power system simulations.

    This class is used to limit the output of a signal within a specified range. It is a basic yet crucial
    component in various control and simulation scenarios.

    Attributes:
        limit (float or torch.Tensor): The limit value for both positive and negative sides.
    """

    def __init__(self, limit, parallel_sims=None):
        """
        Initializes the Limiter block with a specified limit.

        Args:
            limit (float): The limit value for both positive and negative sides.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.limit = limit
        self.enable_parallel_simulation(parallel_sims)

    def get_output(self, input_var):
        """
        Computes the output of the Limiter model.
        Args:
            input_var (torch.Tensor): The current input to the model.

        Returns: The output of the model.

        """
        # noinspection PyTypeChecker
        output = torch.minimum(torch.maximum(input_var.real, -self.limit), self.limit)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.limit = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.limit


class LeadLag(object):
    """
    Represents a Lead-Lag control block in power system simulations.

    This class models a Lead-Lag compensator, which is commonly used in control systems to improve
    the stability and speed of response. It adjusts the phase of a signal and can be used to
    compensate for delays in a control system.

    Attributes:
        t_1 (float or torch.Tensor): Time constant for the lead part of the block.
        t_2 (float or torch.Tensor): Time constant for the lag part of the block.
        input (float or torch.Tensor): Current input to the LeadLag block.
        state_1 (float or torch.Tensor): Internal state of the LeadLag block.
    """

    def __init__(self, t_1, t_2, parallel_sims=None):
        """
        Initializes the LeadLag block with specified parameters.

        Args:
            t_1 (float): Time constant for the lead part of the block.
            t_2 (float): Time constant for the lag part of the block.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.t_1 = t_1
        self.t_2 = t_2
        self.enable_parallel_simulation(parallel_sims)

        self.input = 0
        self.state_1 = 0

    def differential(self):
        """
        Computes the differential equations for the LeadLag block.
        Returns: A tensor containing the derivatives of the state variables.

        """
        dx1 = (1 / self.t_2) * (self.input - self.state_1)
        return torch.stack([dx1, ], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the LeadLag block.
        Returns: The current state vector of the model.

        """
        return torch.stack([self.state_1, ], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the LeadLag block.
        Args:
            x (torch.tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Computes the output of the LeadLag block.
        Args:
            input_var: The current input to the model.

        Returns:

        """
        self.input = input_var
        output = self.t_1 / self.t_2 * input_var + (1 - (self.t_1 / self.t_2)) * self.state_1
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.t_1 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_1
        self.t_2 = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_2

    def initialize(self, out_wish):
        """
        Initializes the LeadLag block with a specified output.
        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.

        """
        self.state_1 = out_wish
        return out_wish


class Washout(object):
    """
    Represents a Washout filter in power system simulations.

    The Washout filter is a high-pass filter that allows signals with frequencies higher than a certain
    threshold to pass through while attenuating lower frequency signals. This filter is often used in
    control systems to isolate dynamic components of a signal.

    Attributes:
        k_w (float or torch.Tensor): Gain of the Washout filter.
        t_w (float or torch.Tensor): Time constant of the Washout filter.
        input (float or torch.Tensor): Current input to the Washout block.
        state_1 (float or torch.Tensor): Internal state of the Washout block.
    """

    def __init__(self, k_w, t_w, parallel_sims=None):
        """
        Initializes the Washout filter with specified parameters.

        Args:
            k_w (float): Gain of the Washout filter.
            t_w (float): Time constant of the Washout filter.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        self.k_w = k_w
        self.t_w = t_w
        self.enable_parallel_simulation(parallel_sims)

        self.input = 0
        self.state_1 = 0

    def differential(self):
        """
        Computes the differential equations for the Washout filter.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        dx1 = 1 / self.t_w * (self.k_w * self.input - self.state_1)
        return torch.stack([dx1, ], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the Washout filter.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.stack([self.state_1, ], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the Washout filter.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.state_1 = x[:, 0]

    def get_output(self, input_var):
        """
        Computes the output of the Washout filter.
        Args:
            input_var: The current input to the model.

        Returns: The output of the model.

        """
        self.input = input_var
        output = 1 / self.t_w * (self.k_w * self.input - self.state_1)
        return output

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations by transforming the model's parameters into tensors.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.k_w = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.k_w
        self.t_w = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.t_w

    def initialize(self, out_wish):
        """
        Initializes the Washout filter with a specified output.
        Args:
            out_wish: The desired output of the model.

        Returns: The desired input to the model.

        """
        self.state_1 = (self.k_w * out_wish)
        return out_wish / self.k_w
