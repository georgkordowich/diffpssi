"""
The model of a generator (Synchronous Machine Model) is defined in this file.
The model is a 6th order differential equation, which is solved during the simulation.
"""
from src.diffpssi.power_sim_lib.backend import *


class SynchMachine(object):
    """
    Represents a Synchronous Machine in power system simulations.

    The SynchMachine class models a synchronous generator or motor, including its dynamic behavior and
    interactions with external control systems such as exciters, governors, and power system stabilizers.

    Attributes:
        s_n, v_n, p_soll_mw, v_soll, h, d, x_d, x_q, x_d_t, x_q_t, x_d_st, x_q_st, t_d0_t, t_q0_t, t_d0_st, t_q0_st:
            Parameters defining the electrical and mechanical characteristics of the machine.
        exciter (SEXS): The exciter model associated with the machine.
        governor (TGOV1): The governor model associated with the machine.
        stabilizer (STAB1): The power system stabilizer model associated with the machine.
    """

    def __init__(self, f_n_sys, s_n_sys, v_n_sys,
                 param_dict=None,
                 name=None,
                 bus=None,
                 s_n=None,
                 v_n=None,
                 p=None,
                 v=None,
                 h=None,
                 d=None,
                 x_d=None,
                 x_q=None,
                 x_d_t=None,
                 x_q_t=None,
                 x_d_st=None,
                 x_q_st=None,
                 t_d0_t=None,
                 t_q0_t=None,
                 t_d0_st=None,
                 t_q0_st=None,
                 ):
        """
        Initializes the SynchMachine object with specified parameters.

        Args:
            param_dict (dict, optional): A dictionary of parameters for the machine.
            name (str, optional): The name of the machine.
            bus (Bus, optional): The bus the machine is connected to.
            s_n (float, optional): The apparent power rating of the machine in MVA.
            v_n (float, optional): The rated voltage of the machine in kV.
            p (float, optional): The active power of the machine in MW.
            v (float, optional): The voltage of the machine in per unit.
            h (float, optional): The inertia constant of the machine in seconds.
            d (float, optional): The damping coefficient of the machine in pu.
            x_d (float, optional): The direct-axis transient reactance of the machine in pu.
            x_q (float, optional): The quadrature-axis transient reactance of the machine in pu.
            x_d_t (float, optional): The direct-axis transient reactance of the machine in pu.
            x_q_t (float, optional): The quadrature-axis transient reactance of the machine in pu.
            x_d_st (float, optional): The direct-axis subtransient reactance of the machine in pu.
            x_q_st (float, optional): The quadrature-axis subtransient reactance of the machine in pu.
            t_d0_t (float, optional): The direct-axis open-circuit time constant of the machine in seconds.
            t_q0_t (float, optional): The quadrature-axis open-circuit time constant of the machine in seconds.
            t_d0_st (float, optional): The direct-axis open-circuit subtransient time constant of the machine in
            seconds.
            t_q0_st (float, optional): The quadrature-axis open-circuit subtransient time constant of the machine in
            seconds.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'bus': bus,
                'S_n': s_n,
                'V_n': v_n,
                'P': p,
                'V': v,
                'H': h,
                'D': d,
                'X_d': x_d,
                'X_q': x_q,
                'X_d_t': x_d_t,
                'X_q_t': x_q_t,
                'X_d_st': x_d_st,
                'X_q_st': x_q_st,
                'T_d0_t': t_d0_t,
                'T_q0_t': t_q0_t,
                'T_d0_st': t_d0_st,
                'T_q0_st': t_q0_st,
            }

        self.parallel_sims = None

        self.name = param_dict['name']
        self.bus = param_dict['bus']
        self.s_n = param_dict['S_n']
        self.v_n = param_dict['V_n']
        self.p_soll_mw = param_dict['P']
        self.v_soll = param_dict['V']
        self.h = param_dict['H']
        self.d = param_dict['D']
        self.x_d = param_dict['X_d']
        self.x_q = param_dict['X_q']
        self.x_d_t = param_dict['X_d_t']
        self.x_q_t = param_dict['X_q_t']
        self.x_d_st = param_dict['X_d_st']
        self.x_q_st = param_dict['X_q_st']
        self.t_d0_t = param_dict['T_d0_t']
        self.t_q0_t = param_dict['T_q0_t']
        self.t_d0_st = param_dict['T_d0_st']
        self.t_q0_st = param_dict['T_q0_st']

        # initialize system vars
        self.fn = f_n_sys
        self.s_n_sys = s_n_sys
        self.v_n_sys = v_n_sys

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

        self.exciter = None
        self.governor = None
        self.stabilizer = None

    def differential(self):
        """
        Computes the differential equations governing the dynamics of the synchronous machine.

        This method calculates the time derivatives of the state variables of the synchronous machine,
        including its interaction with the exciter, governor, and power system stabilizer models if they are present.

        Returns:
            torch.Tensor: A tensor containing the derivatives of the state variables.
        """
        if self.exciter:
            if self.stabilizer:
                pss_output = self.stabilizer.get_output(self.omega)
                self.e_fd = self.exciter.get_output(torch.abs(self.v_bb) - pss_output)
            else:
                self.e_fd = self.exciter.get_output(torch.abs(self.v_bb))

        if self.governor:
            self.p_m = self.governor.get_output(self.omega)

        t_m = self.p_m / (1 + self.omega)
        d_omega = 1 / (2 * self.h) * (t_m - self.p_e - self.d * self.omega)
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
        if self.stabilizer:
            deriv_vec = torch.concatenate((deriv_vec, self.stabilizer.differential()), axis=1)

        return deriv_vec

    def add_exciter(self, exciter_model):
        """
        Associates an exciter model with the synchronous machine.

        Args:
            exciter_model: The exciter model to associate with the machine.
        """
        self.exciter = exciter_model

    def add_governor(self, governor_model):
        """
        Associates a governor model with the synchronous machine.

        Args:
            governor_model: The governor model to associate with the machine.
        """
        self.governor = governor_model

    def add_pss(self, pss_model):
        """
        Associates a power system stabilizer model with the synchronous machine.

        Args:
            pss_model: The power system stabilizer model to associate with the machine.
        """

        self.stabilizer = pss_model

    def set_state_vector(self, x):
        """
        Sets the state vector of the synchronous machine based on provided values.

        Args:
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
            self.governor.set_state_vector(x[:, 6 + offset: 8 + offset])
            offset += 2
        if self.stabilizer:
            self.stabilizer.set_state_vector(x[:, 6 + offset: 9 + offset])
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
        if self.stabilizer:
            state_vec = torch.concatenate((state_vec, self.stabilizer.get_state_vector()), axis=1)

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
        i_inj = (i_d + i_q) * (self.s_n / self.s_n_sys)
        return i_inj

    def update_internal_vars(self, v_bb):
        """
        Updates internal variables of the synchronous machine based on the voltage at the busbar.

        Args:
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

        Args:
            s_calc (float or torch.Tensor): Calculated apparent power.
            v_bb (float or torch.Tensor): Voltage at the busbar.
        """

        # First calculate the currents of the generator at the busbar.
        # Those currents can then be used to calculate all internal voltages.
        i_gen = torch.conj(s_calc / v_bb) / (self.s_n / self.s_n_sys)

        # Calculate the internal voltages and angle of the generator
        # Basically this is always U2 = U1 + jX * I
        e = v_bb + 1j * self.x_q * i_gen

        self.delta = torch.angle(e)
        self.omega = torch.zeros_like(self.delta)

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

        self.p_m = s_calc.real / (self.s_n / self.s_n_sys)
        self.e_fd = self.e_q_t + i_d * (self.x_d - self.x_d_t)

        if self.exciter:
            self.exciter.initialize(self.e_fd)

        if self.governor:
            self.governor.initialize(self.p_m)

        if self.stabilizer:
            self.stabilizer.initialize(self.e_fd * 0)

    def get_admittance(self, dyn):
        """
        Returns the admittance value of the synchronous machine.

        Args:
            dyn (bool): Flag indicating whether to calculate dynamic (True) or static (False) admittance.

        Returns:
            torch.Tensor: Admittance of the synchronous machine.
        """
        if dyn:
            return -1j / self.x_d_st * (self.s_n / self.s_n_sys)
        else:
            return torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

    def get_lf_power(self):
        """
        Retrieves the load flow power of the synchronous machine.

        Returns:
            torch.Tensor: The load flow power.
        """
        return (self.p_soll_mw + 1j * 0) / self.s_n_sys

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables running parallel simulations for the synchronous machine by converting parameters to tensors

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.parallel_sims = parallel_sims

        self.s_n = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.s_n
        self.v_n = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_n
        self.p_soll_mw = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.p_soll_mw
        self.v_soll = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_soll

        self.h = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.h
        self.d = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.d
        self.x_d = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_d
        self.x_q = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_q
        self.x_d_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_d_t
        self.x_q_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_q_t
        self.x_d_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_d_st
        self.x_q_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.x_q_st
        self.t_d0_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.t_d0_t
        self.t_q0_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.t_q0_t
        self.t_d0_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.t_d0_st
        self.t_q0_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.t_q0_st

        self.omega = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.omega
        self.delta = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.delta
        self.e_q_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.e_q_t
        self.e_d_t = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.e_d_t
        self.e_q_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.e_q_st
        self.e_d_st = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.e_d_st

        self.p_m = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.p_m
        self.p_e = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.p_e
        self.e_fd = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.e_fd
        self.i_d = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.i_d
        self.i_q = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.i_q
        self.v_bb = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_bb
