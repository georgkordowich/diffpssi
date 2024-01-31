"""
Contains the STAB1 model for power system stabilizers. Other models can be added here as well.
"""
from diffpssi.power_sim_lib.backend import *
from diffpssi.power_sim_lib.models.blocks import LeadLag, Washout, Limiter


class STAB1(object):
    """
    Represents the STAB1 (Stabilizer) model in power system simulations.

    The STAB1 model simulates a power system stabilizer, which is designed to enhance the damping of
    power system oscillations through modulation of generator excitation. It typically includes washout
    filters, lead-lag compensators, and limiters to effectively damp oscillations.

    Attributes:
        washout (Washout): The Washout filter block for isolating dynamic components of a signal.
        lead_lag1 (LeadLag): The first LeadLag block in the stabilizer.
        lead_lag2 (LeadLag): The second LeadLag block in the stabilizer.
        limiter (Limiter): The Limiter block to restrict the output within a specific range.
    """

    def __init__(self, param_dict=None,
                 name=None,
                 gen=None,
                 k=None,
                 t=None,
                 t_1=None,
                 t_2=None,
                 t_3=None,
                 t_4=None,
                 h_lim=None,
                 ):
        """
        Initializes the STAB1 model with specified parameters.

        Args:
            param_dict (dict, optional): A dictionary of parameters for the model.
            name (str, optional): The name of the model.
            gen (str, optional): The name of the generator the model is connected to.
            k (float, optional): The gain of the washout filter.
            t (float, optional): The time constant of the washout filter.
            t_1 (float, optional): The time constant t1 of the first lead-lag compensator.
            t_2 (float, optional): The time constant t1 of the second lead-lag compensator.
            t_3 (float, optional): The time constant t2 of the first lead-lag compensator.
            t_4 (float, optional): The time constant t2 of the second lead-lag compensator.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'gen': gen,
                'K': k,
                'T': t,
                'T_1': t_1,
                'T_2': t_2,
                'T_3': t_3,
                'T_4': t_4,
                'H_lim': h_lim,
            }
        self.name = param_dict['name']
        self.gen = param_dict['gen']
        self.k_w = param_dict['K']
        self.t_w = param_dict['T']
        self.t_1 = param_dict['T_1']
        self.t_2 = param_dict['T_2']
        self.t_3 = param_dict['T_3']
        self.t_4 = param_dict['T_4']
        self.h_lim = param_dict['H_lim']

        self.washout = Washout(k_w=self.k_w, t_w=self.t_w)
        self.lead_lag1 = LeadLag(t_1=self.t_1, t_2=self.t_3)
        self.lead_lag2 = LeadLag(t_1=self.t_2, t_2=self.t_4)
        self.limiter = Limiter(limit=self.h_lim)

    def differential(self):
        """
        Computes the differential equations for the STAB1 model.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        return torch.concatenate([self.washout.differential(), self.lead_lag1.differential(),
                                  self.lead_lag2.differential()], axis=1)

    def get_state_vector(self):
        """
        Retrieves the current state vector of the STAB1 model.

        Returns:
            torch.Tensor: The current state vector of the model.
        """
        return torch.concatenate([self.washout.get_state_vector(), self.lead_lag1.get_state_vector(),
                                  self.lead_lag2.get_state_vector()], axis=1)

    def set_state_vector(self, x):
        """
        Sets the state vector of the STAB1 model.

        Args:
            x (torch.Tensor): A tensor representing the new state vector.
        """
        self.washout.set_state_vector(x[:, 0:1])
        self.lead_lag1.set_state_vector(x[:, 1:2])
        self.lead_lag2.set_state_vector(x[:, 2:3])

    def get_output(self, omega_diff):
        """
        Computes the output of the STAB1 model given the frequency deviation.

        Args:
            omega_diff (torch.Tensor): The deviation of the system frequency from its nominal value.

        Returns:
            torch.Tensor: The output of the model, representing the stabilizer's response.
        """
        in1 = self.washout.get_output(omega_diff)
        in2 = self.lead_lag1.get_output(in1)
        in3 = self.lead_lag2.get_output(in2)
        out1 = self.limiter.get_output(in3)

        return out1

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations for the STAB1 model.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.washout.enable_parallel_simulation(parallel_sims)
        self.lead_lag1.enable_parallel_simulation(parallel_sims)
        self.lead_lag2.enable_parallel_simulation(parallel_sims)
        self.limiter.enable_parallel_simulation(parallel_sims)

    def initialize(self, v_pss):
        """
        Initializes the STAB1 model for simulation.

        Args:
            v_pss (float or torch.Tensor): The initial value for the PSS voltage.
        """
        # put the values here that shall come out of the blocks in the first time step so that all derivatives are zero
        in_1 = self.lead_lag2.initialize(v_pss)
        in_2 = self.lead_lag1.initialize(in_1)
        self.washout.initialize(in_2)
