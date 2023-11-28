import torch

from power_sim_lib.models.avrs import SEXS
from power_sim_lib.models.backend import *
from power_sim_lib.models.generic_dynamic_model import GenericModel
from power_sim_lib.models.governors import TGOV1
from power_sim_lib.models.stabilizers import STAB1


class SynchMachine(GenericModel):
    """
    Represents a Synchronous Machine in power system simulations.

    The SynchMachine class models a synchronous generator or motor, including its dynamic behavior and
    interactions with external control systems such as exciters, governors, and power system stabilizers.

    Attributes:
        s_n, v_n, p_soll_mw, v_soll, h, d, x_d, x_q, x_d_t, x_q_t, x_d_st, x_q_st, t_d0_t, t_q0_t, t_d0_st, t_q0_st:
            Parameters defining the electrical and mechanical characteristics of the machine.
        exciter (SEXS): The exciter model associated with the machine.
        governor (TGOV1): The governor model associated with the machine.
        pss (STAB1): The power system stabilizer model associated with the machine.
    """
    def __init__(self, param_dict=None, s_n=2200, v_n=24, p_soll_mw=1998, v_soll=1, h=3.5, d=0,
                 x_d=1.81, x_q=1.76, x_d_t=0.3, x_q_t=0.65, x_d_st=0.23, x_q_st=0.23,
                 t_d0_t=8.0, t_q0_t=1, t_d0_st=0.03, t_q0_st=0.07, parallel_sims=None):
        """
        Initializes the SynchMachine object with specified parameters.

        Parameters:
            param_dict (dict, optional): A dictionary of parameters for the machine.
            s_n, v_n, p_soll_mw, v_soll, h, d, x_d, x_q, x_d_t, x_q_t, x_d_st, x_q_st, t_d0_t, t_q0_t, t_d0_st, t_q0_st:
                Electrical and mechanical characteristics of the machine.
            parallel_sims (int, optional): Number of parallel simulations to enable.
        """
        if param_dict is not None:
            # simply take all values from the dictionary and assign them to the object
            self.__dict__.update(param_dict)
        else:
            self.s_n = s_n
            self.v_n = v_n
            self.p_soll_mw = p_soll_mw
            self.v_soll = v_soll
            self.h = h
            self.d = d
            self.x_d = x_d
            self.x_q = x_q
            self.x_d_t = x_d_t
            self.x_q_t = x_q_t
            self.x_d_st = x_d_st
            self.x_q_st = x_q_st
            self.t_d0_t = t_d0_t
            self.t_q0_t = t_q0_t
            self.t_d0_st = t_d0_st
            self.t_q0_st = t_q0_st

        # initialize states
        self.omega = 0
        self.delta = 0
        self.e_q_t = 0
        self.e_d_t = 0
        self.e_q_st = 0
        self.e_d_st = 0

        # initialize internal variables
        self.p_m = 0
        self.p_e = 0
        self.e_fd = 0
        self.i_d = 0
        self.i_q = 0
        self.v_bb = 0

        # initialize system vars
        self.fn = 0
        self.sys_s_n = 0
        self.sys_v_n = 0

        self.exciter = None
        self.governor = None
        self.pss = None

        self.parallel_sims = parallel_sims

        if parallel_sims is not None:
            self.enable_parallel_simulation(parallel_sims)

    def differential(self):
        """
        Computes the differential equations governing the dynamics of the synchronous machine.

        This method calculates the time derivatives of the state variables of the synchronous machine,
        including its interaction with the exciter, governor, and power system stabilizer models if they are present.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        if self.exciter:
            if self.pss:
                pss_output = self.pss.get_output(self.omega)
                self.e_fd = self.exciter.get_output(torch.abs(self.v_bb) - pss_output)
            else:
                self.e_fd = self.exciter.get_output(torch.abs(self.v_bb))

        if self.governor:
            self.p_m = self.governor.get_output(self.omega)

        t_m = self.p_m / (1 + self.omega)
        d_omega = 1 / (2 * self.h) * (t_m - self.p_e)
        d_delta = self.omega * 2 * torch.pi * self.fn
        d_e_q_t = 1 / self.t_d0_t * (self.e_fd - self.e_q_t - self.i_d * (self.x_d - self.x_d_t))
        d_e_d_t = 1 / self.t_q0_t * (-self.e_d_t + self.i_q * (self.x_q - self.x_q_t))
        d_e_q_st = 1 / self.t_d0_st * (self.e_q_t - self.e_q_st - self.i_d * (self.x_d_t - self.x_d_st))
        d_e_d_st = 1 / self.t_q0_st * (self.e_d_t - self.e_d_st + self.i_q * (self.x_q_t - self.x_q_st))

        deriv_vec = torch.stack([d_omega, d_delta, d_e_q_t, d_e_d_t, d_e_q_st, d_e_d_st], axis=1)
        if self.exciter:
            deriv_vec = torch.concatenate((deriv_vec, self.exciter.differential()), axis=1)
        if self.governor:
            deriv_vec = torch.concatenate((deriv_vec, self.governor.differential()), axis=1)
        if self.pss:
            deriv_vec = torch.concatenate((deriv_vec, self.pss.differential()), axis=1)

        return deriv_vec

    def add_exciter(self, exciter_dict):
        """
        Associates an exciter model with the synchronous machine.

        Parameters:
            exciter_dict (dict): A dictionary containing parameters for initializing the exciter model.
        """
        self.exciter = SEXS(param_dict=exciter_dict, parallel_sims=self.parallel_sims)

    def add_governor(self, governor_dict):
        """
        Associates a governor model with the synchronous machine.

        Parameters:
            governor_dict (dict): A dictionary containing parameters for initializing the governor model.
        """
        self.governor = TGOV1(param_dict=governor_dict, parallel_sims=self.parallel_sims)

    def add_pss(self, pss_dict):
        """
        Associates a power system stabilizer model with the synchronous machine.

        Parameters:
            pss_dict (dict): A dictionary containing parameters for initializing the power system stabilizer model.
        """

        self.pss = STAB1(param_dict=pss_dict, parallel_sims=self.parallel_sims)

    def set_state_vector(self, x):
        """
        Sets the state vector of the synchronous machine based on provided values.

        Parameters:
            x (torch.Tensor): A tensor representing the new state vector.
        """

        self.omega = x[:, 0]
        self.delta = x[:, 1]
        self.e_q_t = x[:, 2]
        self.e_d_t = x[:, 3]
        self.e_q_st = x[:, 4]
        self.e_d_st = x[:, 5]

        offset = 0
        if self.exciter:
            self.exciter.set_state_vector(x[:, 6:8])
            offset += 2
        if self.governor:
            self.governor.set_state_vector(x[:, 6 + offset : 8 + offset])
            offset += 2
        if self.pss:
            self.pss.set_state_vector(x[:, 6 + offset: 9 + offset])
            offset += 3

    def get_state_vector(self):
        """
        Retrieves the current state vector of the synchronous machine.

        Returns:
            torch.Tensor: The current state vector of the machine.
        """

        state_vec = torch.stack([self.omega, self.delta, self.e_q_t, self.e_d_t, self.e_q_st, self.e_d_st], axis=1)
        if self.exciter:
            state_vec = torch.concatenate((state_vec, self.exciter.get_state_vector()), axis=1)
        if self.governor:
            state_vec = torch.concatenate((state_vec, self.governor.get_state_vector()), axis=1)
        if self.pss:
            state_vec = torch.concatenate((state_vec, self.pss.get_state_vector()), axis=1)

        return state_vec

    def calc_current_injections(self):
        """
        Calculates the current injections from the synchronous machine into the network.

        Returns:
            torch.Tensor: A tensor representing the current injections at the machine's terminals.
        """

        i_d = self.e_q_st / (1j * self.x_d_st) * torch.exp(1j * self.delta)
        i_q = self.e_d_st / (1j * self.x_q_st) * torch.exp(1j * (self.delta - torch.pi / 2))

        # transform it to the base system. sn/vn is local per unit and sys_s_n/sys_v_n is global per unit
        i_inj = (i_d + i_q)*(self.s_n/self.sys_s_n)
        return i_inj

    def update_internal_vars(self, v_bb):
        """
        Updates internal variables of the synchronous machine based on the voltage at the busbar.

        Parameters:
            v_bb (float or torch.Tensor): Voltage at the busbar.
        """

        self.v_bb = v_bb

        e_st = self.e_q_st * torch.exp(1j * self.delta) + self.e_d_st * torch.exp(1j * (self.delta - torch.pi / 2))
        self.p_e = (v_bb * torch.conj((e_st - v_bb) / (1j * self.x_d_st))).real
        i_gen = (e_st - v_bb) / (1j * self.x_d_st)
        self.i_d = (i_gen * torch.exp(1j * (torch.pi / 2 - self.delta))).real
        self.i_q = (i_gen * torch.exp(1j * (torch.pi / 2 - self.delta))).imag

    def initialize(self, s_calc, v_bb):
        """
        Initializes the synchronous machine's states based on provided conditions.

        Parameters:
            s_calc (float or torch.Tensor): Calculated apparent power.
            v_bb (float or torch.Tensor): Voltage at the busbar.
        """

        # First calculate the currents of the generator at the busbar.
        # Those currents can then be used to calculate all internal voltages.
        i_gen = torch.conj(s_calc / v_bb)/(self.s_n / self.sys_s_n)

        # Calculate the internal voltages and angle of the generator
        # Basically this is always U2 = U1 + jX * I
        e = v_bb + 1j * self.x_q * i_gen
        self.delta = torch.angle(e)

        i_dq = i_gen * torch.exp(1j * (torch.pi / 2 - self.delta))
        i_d = i_dq.real
        i_q = i_dq.imag  # q-axis leading d-axis

        v_g_dq = v_bb * torch.exp(1j * (torch.pi / 2 - self.delta))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        self.e_q_t = v_q + self.x_d_t * i_d
        self.e_d_t = v_d - self.x_q_t * i_q

        self.e_q_st = v_q + self.x_d_st * i_d
        self.e_d_st = v_d - self.x_q_st * i_q

        self.p_m = s_calc.real/(self.s_n / self.sys_s_n)
        self.e_fd = self.e_q_t + i_d * (self.x_d - self.x_d_t)

        if self.exciter:
            self.exciter.initialize(self.e_fd)

        if self.governor:
            self.governor.initialize(self.p_m)

        if self.pss:
            self.pss.initialize(self.e_fd * 0)

    def set_sys_vars(self, fn, base_mva, base_voltage):
        """
        Sets system-level variables like system frequency and base power.

        Parameters:
            fn (float): System frequency.
            base_mva (float): Base power in MVA.
            base_voltage (float): Base voltage.
        """

        self.fn = fn
        self.sys_s_n = base_mva
        self.sys_v_n = base_voltage

    def get_admittance(self, dyn):
        """
        Returns the admittance value of the synchronous machine.

        Parameters:
            dyn (bool): Flag indicating whether to calculate dynamic (True) or static (False) admittance.

        Returns:
            torch.Tensor: Admittance of the synchronous machine.
        """
        if dyn:
            return -1j/self.x_d_st * (self.s_n / self.sys_s_n)
        else:
            return torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

    def get_lf_power(self):
        """
        Retrieves the load flow power of the synchronous machine.

        Returns:
            torch.Tensor: The load flow power.
        """
        return (self.p_soll_mw + 1j * 0)/self.sys_s_n

    def enable_parallel_simulation(self, parallel_sims):
        def enable_parallel_simulation(self, parallel_sims):
            """
            Enables running parallel simulations for the synchronous machine by converting parameters to tensors

            Parameters:
                parallel_sims (int): Number of parallel simulations.
            """

        self.s_n = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.s_n
        self.v_n = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.v_n
        self.p_soll_mw = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.p_soll_mw
        self.v_soll = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.v_soll

        self.h = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.h
        self.d = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.d
        self.x_d = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_d
        self.x_q = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_q
        self.x_d_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_d_t
        self.x_q_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_q_t
        self.x_d_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_d_st
        self.x_q_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.x_q_st
        self.t_d0_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.t_d0_t
        self.t_q0_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.t_q0_t
        self.t_d0_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.t_d0_st
        self.t_q0_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.t_q0_st

        self.omega = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.omega
        self.delta = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.delta
        self.e_q_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.e_q_t
        self.e_d_t = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.e_d_t
        self.e_q_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.e_q_st
        self.e_d_st = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.e_d_st

        self.p_m = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.p_m
        self.p_e = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.p_e
        self.e_fd = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.e_fd
        self.i_d = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.i_d
        self.i_q = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.i_q
        self.v_bb = torch.ones((parallel_sims, 1), dtype=torch.complex128)*self.v_bb

    def reset(self):
        def reset(self):
            """
            Resets the states and internal variables of the synchronous machine to their default values.
            """
        self.omega = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.delta = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.e_q_t = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.e_d_t = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.e_q_st = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.e_d_st = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

        self.p_m = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.p_e = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.e_fd = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.i_d = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.i_q = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        self.v_bb = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)


if __name__ == '__main__':
    parameter_dict = {'s_n_gen': 2200, 'v_n_gen': 24, 'p_gen': 1998, 'v_soll_gen': 1, 'h_gen': 3.5, 'd_gen': 0,
                        'x_d_gen': 1.81, 'x_q_gen': 1.76, 'x_d_t_gen': 0.3, 'x_q_t_gen': 0.65, 'x_d_st_gen': 0.23,
                        'x_q_st_gen': 0.23, 't_d0_t_gen': 8.0, 't_q0_t_gen': 1, 't_d0_st_gen': 0.03, 't_q0_st_gen': 0.07}
    gen = SynchMachine('gen1', param_dict=parameter_dict)
    print(gen.s_n_gen)