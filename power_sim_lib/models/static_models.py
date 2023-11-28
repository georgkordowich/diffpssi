import torch

from power_sim_lib.models.backend import *


class ScEvent(object):
    """
    Represents a short circuit event in a power system simulation.

    This class models a short circuit event at a specific bus within a specified time window,
    allowing the simulation of transient conditions and system response to faults.

    Attributes:
        start_time (float): The start time of the short circuit event.
        end_time (float): The end time of the short circuit event.
        bus (int): The index of the bus where the short circuit occurs.
    """
    def __init__(self, start_time, end_time, bus):
        """
        Initializes the ScEvent object with the start time, end time, and bus index.

        Parameters:
            start_time (float): The start time of the short circuit event.
            end_time (float): The end time of the short circuit event.
            bus (int): The index of the bus where the short circuit occurs.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.bus = bus

    def is_active(self, t):
        """
        Checks if the short circuit event is active at a given time.

        Parameters:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        if self.start_time <= t <= self.end_time:
            return True
        else:
            return False


class Bus(object):
    """
    Represents a bus in the power system simulation.

    A bus is a node at which power system components such as generators, loads, and lines are connected.
    This class models the electrical behavior of the bus and interactions with connected components.

    Attributes:
        name (str): The name of the bus.
        lf_type (str): The load flow type of the bus.
        v_n (torch.Tensor): Nominal voltage at the bus.
        models (list): List of models (generators, loads, etc.) connected to the bus.
        voltage (torch.Tensor): Current voltage at the bus.
    """
    def __init__(self, name, lf_type, v_n, parallel_sims=1):
        """
        Initializes the Bus object with the given name, load flow type, and nominal voltage.

        Parameters:
            name (str): The name of the bus.
            lf_type (str): The load flow type of the bus.
            v_n (float): Nominal voltage at the bus.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.parallel_sims = parallel_sims
        self.name = name
        self.models = []
        self.voltage = torch.ones((parallel_sims, 1), dtype=torch.complex128)
        self.lf_type = lf_type
        self.v_n = torch.ones((parallel_sims, 1), dtype=torch.float64) * v_n

    def get_current_injections(self):
        """
        Calculates the total current injection at the bus from all connected models.

        Returns:
            torch.Tensor: The total current injection at the bus.
        """
        current_inj = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        for model in self.models:
            current_inj += model.calc_current_injections()
        return current_inj

    def update_voltages(self, v_bb):
        """
        Updates the voltage at the bus and propagates the update to all connected models.

        Parameters:
            v_bb (torch.Tensor): The new busbar voltage value.
        """
        self.voltage = v_bb

        for model in self.models:
            model.update_internal_vars(v_bb)

    def add_model(self, model):
        """
        Adds a power system model (e.g., generator, load) to the bus.

        Parameters:
            model (GenericModel): The model to add to the bus.
        """
        self.models.append(model)

    def get_lf_power(self):
        """
        Calculates the total load flow power at the bus from all connected models.

        Returns:
            torch.Tensor: The total load flow power at the bus.
        """
        s_soll = torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)
        for model in self.models:
            s_soll += model.get_lf_power()
        return s_soll

    def reset(self):
        """
        Resets the bus to its initial state.
        """
        self.voltage = torch.ones((self.parallel_sims, 1), dtype=torch.complex128)


class Line(object):
    """
    Represents a transmission line in the power system simulation.

    This class models the electrical characteristics of a transmission line, including resistance,
    reactance, and susceptance, between two buses in the power system.

    Attributes:
        from_bus (int): The index of the starting bus of the line.
        to_bus (int): The index of the ending bus of the line.
        r (torch.Tensor): Resistance of the line.
        x (torch.Tensor): Reactance of the line.
        b (torch.Tensor): Susceptance of the line.
    """

    def __init__(self, from_bus, to_bus, r, x, b, length, unit, s_n, v_n, parallel_sims=None):
        """
        Initializes the Line object with the specified electrical parameters and connected buses.

        Parameters:
            from_bus (int): The index of the starting bus of the line.
            to_bus (int): The index of the ending bus of the line.
            r (float): Resistance of the line.
            x (float): Reactance of the line.
            b (float): Susceptance of the line.
            length (float): Length of the line.
            unit (str): Unit of the parameters ('Ohm' or 'pu').
            s_n (float): Base power in MVA for per-unit conversion.
            v_n (float): Base voltage for per-unit conversion.
            parallel_sims (int, optional): Number of parallel simulations.
        """
        self.from_bus = from_bus
        self.to_bus = to_bus

        z_n = v_n ** 2 / s_n

        if unit == 'Ohm':
            self.r = r/z_n * length
            self.x = x/z_n * length
            self.b = b * z_n * length
        elif unit == 'pu':
            self.r = r * length
            self.x = x * length
            self.b = b * length
        else:
            raise ValueError('Unit not supported')

        if parallel_sims is not None:
            self.enable_parallel_simulation(parallel_sims)

    def get_admittance_diagonal(self):
        """
        Calculates the diagonal admittance value of the line.

        Returns:
            torch.Tensor: Diagonal admittance of the line.
        """
        return 1 / (self.r + 1j * self.x) + 1j * self.b / 2

    def get_admittance_off_diagonal(self):
        """
        Calculates the off-diagonal admittance value of the line.

        Returns:
            torch.Tensor: Off-diagonal admittance of the line.
        """
        return -1 / (self.r + 1j * self.x)

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations for the line.

        Parameters:
            parallel_sims (int): Number of parallel simulations.
        """
        self.from_bus = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.from_bus
        self.to_bus = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.to_bus
        self.r = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.r
        self.x = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.x
        self.b = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.b


class Transfomer(object):
    """
    Represents a transformer in the power system simulation.

    This class models the electrical characteristics of a transformer, including resistance,
    reactance, and connection between two buses in the power system.

    Attributes:
        from_bus (int): The index of the primary side bus of the transformer.
        to_bus (int): The index of the secondary side bus of the transformer.
        r (torch.Tensor): Resistance of the transformer.
        x (torch.Tensor): Reactance of the transformer.
        b (torch.Tensor): Susceptance of the transformer (usually zero).
    """
    def __init__(self, from_bus, to_bus, s_n, r, x, v_n_from, v_n_to, s_n_sys, parallel_sims=None):
        """
        Initializes the Transformer object with the specified electrical parameters and connected buses.

        Parameters:
            from_bus (int): The index of the primary side bus of the transformer.
            to_bus (int): The index of the secondary side bus of the transformer.
            s_n (float): Rated power of the transformer.
            r (float): Resistance of the transformer.
            x (float): Reactance of the transformer.
            v_n_from (float): Nominal voltage of the primary side.
            v_n_to (float): Nominal voltage of the secondary side.
            s_n_sys (float): Base power in MVA for the system.
            parallel_sims (int, optional): Number of parallel simulations.
        """
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.s_n = s_n
        self.r = r
        self.x = x
        self.b = 0
        self.v_n_from = v_n_from
        self.v_n_to = v_n_to

        self.s_n_sys = s_n_sys

        if parallel_sims is not None:
            self.enable_parallel_simulation(parallel_sims)

    def get_admittance_diagonal(self):
        """
        Calculates the diagonal admittance value of the transformer.

        Returns:
            torch.Tensor: Diagonal admittance of the transformer.
        """
        return (1 / (self.r + 1j * self.x) + 1j * self.b / 2)*self.s_n/self.s_n_sys

    def get_admittance_off_diagonal(self):
        """
        Calculates the off-diagonal admittance value of the transformer.

        Returns:
            torch.Tensor: Off-diagonal admittance of the transformer.
        """
        return -1 / (self.r + 1j * self.x)*self.s_n/self.s_n_sys

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations for the transformer.

        Parameters:
            parallel_sims (int): Number of parallel simulations.
        """
        self.from_bus = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.from_bus
        self.to_bus = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.to_bus
        self.r = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.r
        self.x = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.x
        self.v_n_from = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_from
        self.v_n_to = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.v_n_to


class Load(object):
    """
    Represents a load in the power system simulation.

    This class models an electrical load, characterized by its active and reactive power demand,
    at a bus in the power system.

    Attributes:
        p_soll_mw (float): Desired active power (MW) of the load.
        q_soll_mvar (float): Desired reactive power (MVAR) of the load.
    """
    def __init__(self, p_soll_mw, q_soll_mvar, sys_s_n, parallel_sims=None):
        """
        Initializes the Load object with specified active and reactive power demands.

        Parameters:
            p_soll_mw (float): Desired active power (MW) of the load.
            q_soll_mvar (float): Desired reactive power (MVAR) of the load.
            sys_s_n (float): Base power in MVA of the system.
            parallel_sims (int, optional): Number of parallel simulations.
        """
        self.p_soll_mw = p_soll_mw
        self.q_soll_mvar = q_soll_mvar
        self.y_load = None
        self.sys_s_n = sys_s_n

        self.parallel_sims = parallel_sims

        if parallel_sims is not None:
            self.enable_parallel_simulation(parallel_sims)

    def get_lf_power(self):
        """
        Calculates and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power (sum of active and reactive power)
        of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """

        return -(self.p_soll_mw + 1j * self.q_soll_mvar)/self.sys_s_n

    def get_admittance(self, dyn):
        """
        Computes and returns the admittance of the Load.

        The admittance is computed based on dynamic or static analysis, determined by the 'dyn' parameter.
        If dynamic analysis is selected, the existing load admittance value is returned. Otherwise,
        a zero admittance tensor is returned for static analysis.

        Parameters:
            dyn (bool): Flag indicating whether dynamic (True) or static (False) analysis is to be used.

        Returns:
            torch.Tensor: The calculated admittance tensor.
        """

        if dyn:
            return self.y_load
        else:
            return torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

    def initialize(self, s_calc, v_bb):
        """
        Initializes the Load by calculating its admittance.

        This method computes the Load's admittance based on the calculated complex power and the voltage base values.

        Parameters:
            s_calc (float): The calculated complex power.
            v_bb (float): Base voltage value.
        """

        s_load = (self.p_soll_mw + 1j * self.q_soll_mvar)/self.sys_s_n
        z_load = torch.conj(torch.abs(v_bb) ** 2 / s_load)
        self.y_load = 1 / z_load

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulation for the Load.

        This method sets up the load for parallel simulations by initializing the active and reactive power demands
        as tensors with dimensions corresponding to the number of parallel simulations.

        Parameters:
            parallel_sims (int): The number of parallel simulations to enable.
        """

        self.p_soll_mw = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.p_soll_mw
        self.q_soll_mvar = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.q_soll_mvar

    def calc_current_injections(self):
        """
        Calculates and returns the current injections for the Load.

        This method is used in parallel simulations to compute the current injections. It currently returns a zero tensor.

        Returns:
            torch.Tensor: A tensor of zero current injections for the parallel simulations.
        """

        return torch.zeros((self.parallel_sims, 1), dtype=torch.complex128)

    def reset(self):
        """
        Resets the internal state of the Load.

        This method is a placeholder for resetting any internal variables or states of the Load object, as required.
        """

        pass

    def update_internal_vars(self, v_bb):
        """
        Updates internal variables of the Load based on the given voltage base values.

        This method is a placeholder for updating any internal variables of the Load object, based on the provided voltage base values.

        Parameters:
            v_bb (float): Base voltage value.
        """

        pass
