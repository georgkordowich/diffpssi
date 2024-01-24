"""
This module contains classes for modeling static power system components like loads, lines, and transformers.
Other components can be added here as well. All components do not contain state variables and are therefore
modeled as static components.
"""
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

        Args:
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

        Args:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        if self.start_time < t <= self.end_time:
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
    def __init__(self, param_dict, parallel_sims):
        """
        Initializes the Bus object with the given name, load flow type, and nominal voltage.

        Args:
            param_dict (dict): Dictionary of parameters for the bus.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.parallel_sims = parallel_sims
        self.name = param_dict['name']
        self.v_n = param_dict['V_n']

        self.models = []
        self.lf_type = 'PQ'
        self.voltage = torch.ones((parallel_sims, 1), dtype=torch.complex128)

        self.enable_parallel_simulation(parallel_sims)

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

        Args:
            v_bb (torch.Tensor): The new busbar voltage value.
        """
        self.voltage = v_bb

        for model in self.models:
            model.update_internal_vars(v_bb)

    def add_model(self, model):
        """
        Adds a power system model (e.g., generator, load) to the bus.

        Args:
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
        try:
            self.update_voltages(self.models[-1].v_soll)
        # except index and attribute error if no models are connected
        except (IndexError, AttributeError):
            self.update_voltages(torch.ones((self.parallel_sims, 1), dtype=torch.complex128))

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations for the bus.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.v_n = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.v_n


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

    def __init__(self, param_dict, bus_from, bus_to, sys_mva, sys_voltage, parallel_sims):
        """
        Initializes the Line object with the specified electrical parameters and connected buses.

        Args:
            param_dict (dict): Dictionary of parameters for the line.
            bus_from (int): The index of the starting bus of the line.
            bus_to (int): The index of the ending bus of the line.
            sys_mva (float): Base power in MVA for the system.
            sys_voltage (float): Base voltage in kV for the system.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.name = param_dict['name']
        self.from_bus = bus_from
        self.to_bus = bus_to

        length = param_dict['length']
        s_n = param_dict.get('S_n', sys_mva)
        v_n = param_dict.get('V_n', sys_voltage)
        unit = param_dict['unit']
        r = param_dict['R']
        x = param_dict['X']
        b = param_dict['B']

        s_n_sys = sys_mva
        v_n_sys = sys_voltage

        z_n_sys = v_n_sys ** 2 / s_n_sys
        z_n = v_n ** 2 / s_n

        if unit == 'Ohm':
            self.r = r * length / z_n_sys
            self.x = x * length / z_n_sys
            self.b = b * length * z_n_sys
        elif unit == 'p.u.':
            self.r = r * z_n / z_n_sys * length
            self.x = x * z_n / z_n_sys * length
            self.b = b / z_n * z_n_sys * length
        else:
            raise ValueError('Unit not supported')

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

        Args:
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
    def __init__(self, param_dict, bus_from, bus_to, s_n_sys, parallel_sims):
        """
        Initializes the Transformer object with the specified electrical parameters and connected buses.

        Args:
            param_dict (dict): Dictionary of parameters for the transformer.
            bus_from (int): The index of the primary side bus of the transformer.
            bus_to (int): The index of the secondary side bus of the transformer.
            s_n_sys (float): Base power in MVA for the system.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.name = param_dict['name']
        self.from_bus = bus_from
        self.to_bus = bus_to

        self.s_n = param_dict['S_n']
        self.r = param_dict['R']
        self.x = param_dict['X']
        self.v_n_from = param_dict['V_n_from']
        self.v_n_to = param_dict['V_n_to']
        self.s_n_sys = s_n_sys
        self.b = param_dict.get('B', 0)

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

        Args:
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
    def __init__(self, param_dict, s_n_sys, v_n_sys, parallel_sims):
        """
        Initializes the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict): Dictionary of parameters for the load.
            s_n_sys (float): Base power in MVA for the system.
            v_n_sys (float): Base voltage in kV for the system.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.name = param_dict['name']
        self.p_soll_mw = param_dict['P']
        self.q_soll_mvar = param_dict['Q']
        self.model = param_dict['model']

        if not self.model == 'Z':
            raise ValueError('Only Z model is supported for loads')

        self.s_n_sys = s_n_sys

        self.y_load = None

        self.enable_parallel_simulation(parallel_sims)

    def get_lf_power(self):
        """
        Calculates and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power (sum of active and reactive power)
        of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """

        return -(self.p_soll_mw + 1j * self.q_soll_mvar)/self.s_n_sys

    def get_admittance(self, dyn):
        """
        Computes and returns the admittance of the Load.

        The admittance is computed based on dynamic or static analysis, determined by the 'dyn' parameter.
        If dynamic analysis is selected, the existing load admittance value is returned. Otherwise,
        a zero admittance tensor is returned for static analysis.

        Args:
            dyn (bool): Flag indicating whether dynamic (True) or static (False) analysis is to be used.

        Returns:
            torch.Tensor: The calculated admittance tensor.
        """

        if dyn:
            return self.y_load
        else:
            return torch.zeros_like(self.p_soll_mw)

    def initialize(self, s_calc, v_bb):
        """
        Initializes the Load by calculating its admittance.

        This method computes the Load's admittance based on the calculated complex power and the voltage base values.

        Args:
            s_calc (float): The calculated complex power.
            v_bb (torch.Tensor): Busbar voltage value.
        """
        s_load = (self.p_soll_mw + 1j * self.q_soll_mvar)/self.s_n_sys
        z_load = torch.conj(torch.abs(v_bb) ** 2 / s_load)
        self.y_load = 1 / z_load

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulation for the Load.

        This method sets up the load for parallel simulations by initializing the active and reactive power demands
        as tensors with dimensions corresponding to the number of parallel simulations.

        Args:
            parallel_sims (int): The number of parallel simulations to enable.
        """

        self.p_soll_mw = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.p_soll_mw
        self.q_soll_mvar = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.q_soll_mvar

    def calc_current_injections(self):
        """
        Calculates and returns the current injections for the Load.

        This method is used in parallel simulations to compute the current injections.
        It currently returns a zero tensor.

        Returns:
            torch.Tensor: A tensor of zero current injections for the parallel simulations.
        """

        return torch.zeros_like(self.p_soll_mw)

    def update_internal_vars(self, v_bb):
        """
        Updates internal variables of the Load based on the given voltage base values.

        This method is a placeholder for updating any internal variables of the Load object,
        based on the provided voltage base values.

        Args:
            v_bb (float): Base voltage value.
        """
        pass


class Shunt(object):
    """
    Represents a load in the power system simulation.

    This class models an electrical load, characterized by its active and reactive power demand,
    at a bus in the power system.

    Attributes:
        q_soll_mvar (float): Desired reactive power (MVAR) of the load.
    """
    def __init__(self, param_dict, s_n_sys, v_n_sys, parallel_sims):
        """
        Initializes the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict): Dictionary of parameters for the load.
            s_n_sys (float): Base power in MVA for the system.
            v_n_sys (float): Base voltage in kV for the system.
            parallel_sims (int): Number of parallel simulations to run.
        """
        self.name = param_dict['name']
        self.v_n = param_dict['V_n']
        self.q_soll_mvar = param_dict['Q']
        self.model = param_dict['model']

        if not self.model == 'Z':
            raise ValueError('Only Z model is supported for loads')

        self.s_n_sys = s_n_sys

        s_shunt = torch.tensor(-1j * self.q_soll_mvar / self.s_n_sys)
        z = torch.conj(1 / s_shunt)
        self.y_shunt = 1 / z

        self.enable_parallel_simulation(parallel_sims)

    def get_lf_power(self):
        """
        Calculates and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power (sum of active and reactive power)
        of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """

        return torch.zeros_like(self.q_soll_mvar)

    def get_admittance(self, dyn):
        """
        Computes and returns the admittance of the Load.

        The admittance is computed based on dynamic or static analysis, determined by the 'dyn' parameter.
        Currently, the same admittance is returned for both dynamic and static analysis.

        Args:
            dyn (bool): Flag indicating whether dynamic (True) or static (False) analysis is to be used.

        Returns:
            torch.Tensor: The calculated admittance tensor.
        """
        # parameter dyn is not used here, but is required for compatibility with other models
        return self.y_shunt

    def initialize(self, s_calc, v_bb):
        """
        Initializes the Load (theoretically). Currently not used.
        Args:
            s_calc: The calculated complex power.
            v_bb: Voltage at the busbar.
        """
        pass

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulation for the Load.

        This method sets up the load for parallel simulations by initializing the active and reactive power demands
        as tensors with dimensions corresponding to the number of parallel simulations.

        Args:
            parallel_sims (int): The number of parallel simulations to enable.
        """
        self.q_soll_mvar = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.q_soll_mvar
        self.y_shunt = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.y_shunt

    def calc_current_injections(self):
        """
        Calculates and returns the current injections for the Load.

        This method is used in parallel simulations to compute the current injections.
        It currently returns a zero tensor.

        Returns:
            torch.Tensor: A tensor of zero current injections for the parallel simulations.
        """

        return torch.zeros_like(self.q_soll_mvar)

    def update_internal_vars(self, v_bb):
        """
        Updates internal variables of the Load based on the given voltage at the busbar (theoretically).
        Currently not used and only for compatibility with other models.

        Args:
            v_bb (float): Busbar voltage.
        """
        pass
