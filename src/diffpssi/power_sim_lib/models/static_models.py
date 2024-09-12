"""
This module contains classes for modeling static power system components like loads, lines, and transformers.
Other components can be added here as well. All components do not contain state variables and are therefore
modeled as static components.
"""
import torch
from src.diffpssi.power_sim_lib.backend import *


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


class ParamEvent(object):
    """
    Represents a parameter event in a power system simulation.

    This class models a parameter event that changes the value of a parameter for a specific component
    within a specified time window, allowing the simulation of dynamic parameter changes in the system.

    Attributes:
        start_time (float): The start time of the parameter event.
        end_time (float): The end time of the parameter event.
        component (object): The component to which the parameter change applies.
        parameter (str): The name of the parameter to change.
        value (float): The new value of the parameter.
    """

    def __init__(self, start_time, model, param_name, value):
        """
        Initializes the ParameterEvent object with the start time, end time, component, parameter, and value.

        Args:
            start_time (float): The start time of the parameter event.
            parameter (str): The parameter to change
            value (float): The new value of the parameter.
        """
        self.start_time = start_time
        self.model = model
        self.param_name = param_name
        self.value = value
        self.handled = False

    def handle_event(self, t):
        """
        Checks if the parameter event is active at a given time.

        Args:
            t (float): The time at which to check the event's activity.

        Returns:
            bool: True if the event is active at time t, False otherwise.
        """
        if not self.handled and t >= self.start_time:
            # set the attribute by name
            setattr(self.model, self.param_name, self.value)
            self.handled = True




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

    def __init__(self, param_dict=None,
                 name=None,
                 v_n=None):
        """
        Initializes the Bus object with the given name, load flow type, and nominal voltage.

        Args:
            param_dict (dict): Dictionary of parameters for the bus.
            name (str): The name of the bus.
            v_n (float): Nominal voltage at the bus in kV.
        """
        self.parallel_sims = None
        if param_dict is None:
            param_dict = {
                'name': name,
                'V_n': v_n,
            }

        self.name = param_dict['name']
        self.v_n = param_dict['V_n']

        self.models = []
        self.diff_models = []
        self.lf_type = 'PQ'
        self.voltage = 1.0

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
        if hasattr(model, 'differential'):
            self.diff_models.append(model)


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
        self.voltage = torch.ones((parallel_sims, 1), dtype=torch.complex128) * self.voltage
        self.parallel_sims = parallel_sims


class Line(object):
    """
    Represents a transmission line in the power system simulation.

    This class models the electrical characteristics of a transmission line, including resistance,
    reactance, and susceptance, between two buses in the power system.

    Attributes:
        from_bus_id (int): The index of the starting bus of the line.
        to_bus_id (int): The index of the ending bus of the line.
        r (torch.Tensor): Resistance of the line.
        x (torch.Tensor): Reactance of the line.
        b (torch.Tensor): Susceptance of the line.
    """

    def __init__(self, s_n_sys, v_n_sys,
                 param_dict=None,
                 name=None,
                 from_bus=None,
                 to_bus=None,
                 length=None,
                 s_n=None,
                 v_n=None,
                 unit=None,
                 r=None,
                 x=None,
                 b=None):
        """
        Initializes the Line object with the specified electrical parameters and connected buses.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the line.
            name (str, optional): The name of the line.
            from_bus (str, optional): The name of the starting bus of the line.
            to_bus (str, optional): The name of the ending bus of the line.
            length (float, optional): The length of the line in km.
            s_n (float, optional): The nominal power of the line in MVA.
            v_n (float, optional): The nominal voltage of the line in kV.
            unit (str, optional): The unit of the line parameters. Either 'Ohm' or 'p.u.'.
            r (float, optional): The resistance of the line (either in Ohm or p.u.)/length.
            x (float, optional): The reactance of the line (either in Ohm or p.u.)/length.
            b (float, optional): The susceptance of the line (either in Ohm or p.u.)/length.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'from_bus': from_bus,
                'to_bus': to_bus,
                'length': length,
                'S_n': s_n,
                'V_n': v_n,
                'unit': unit,
                'R': r,
                'X': x,
                'B': b,
            }
        self.name = param_dict['name']
        self.from_bus_name = param_dict['from_bus']
        self.to_bus_name = param_dict['to_bus']

        self.from_bus_id = None
        self.to_bus_id = None

        length = param_dict['length']
        s_n = param_dict.get('S_n', s_n_sys)
        v_n = param_dict.get('V_n', v_n_sys)
        unit = param_dict['unit']
        r = param_dict['R']
        x = param_dict['X']
        b = param_dict['B']

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
        self.from_bus_id = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.from_bus_id
        self.to_bus_id = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.to_bus_id
        self.r = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.r
        self.x = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.x
        self.b = torch.ones((parallel_sims, 1), dtype=torch.float64) * self.b


class Transformer(object):
    """
    Represents a transformer in the power system simulation.

    This class models the electrical characteristics of a transformer, including resistance,
    reactance, and connection between two buses in the power system.

    Attributes:
        from_bus_id (int): The index of the primary side bus of the transformer.
        to_bus_id (int): The index of the secondary side bus of the transformer.
        r (torch.Tensor): Resistance of the transformer in p.u.
        x (torch.Tensor): Reactance of the transformer in p.u.
        b (torch.Tensor): Susceptance of the transformer in p.u.
    """

    def __init__(self, s_n_sys,
                 param_dict=None,
                 name=None,
                 from_bus=None,
                 to_bus=None,
                 s_n=None,
                 r=None,
                 x=None,
                 v_n_from=None,
                 v_n_to=None,
                 b=0):
        """
        Initializes the Transformer object with the specified electrical parameters and connected buses.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the transformer.
            name (str, optional): The name of the transformer.
            from_bus (str, optional): The name of the primary side bus of the transformer.
            to_bus (str, optional): The name of the secondary side bus of the transformer.
            s_n (float, optional): The nominal power of the transformer in MVA.
            r (float, optional): The resistance of the transformer in p.u.
            x (float, optional): The reactance of the transformer in p.u.
            v_n_from (float, optional): The nominal voltage of the primary side of the transformer in kV.
            v_n_to (float, optional): The nominal voltage of the secondary side of the transformer in kV.
            b (float, optional): The susceptance of the transformer in p.u.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'from_bus': from_bus,
                'to_bus': to_bus,
                'S_n': s_n,
                'R': r,
                'X': x,
                'V_n_from': v_n_from,
                'V_n_to': v_n_to,
                'B': b,
            }
        self.name = param_dict['name']
        self.from_bus_name = param_dict['from_bus']
        self.to_bus_name = param_dict['to_bus']

        self.from_bus_id = None
        self.to_bus_id = None

        self.s_n = param_dict['S_n']
        self.r = param_dict['R']
        self.x = param_dict['X']
        self.v_n_from = param_dict['V_n_from']
        self.v_n_to = param_dict['V_n_to']
        self.b = param_dict.get('B', 0)

        self.s_n_sys = s_n_sys

    def get_admittance_diagonal(self):
        """
        Calculates the diagonal admittance value of the transformer.

        Returns:
            torch.Tensor: Diagonal admittance of the transformer.
        """
        return (1 / (self.r + 1j * self.x) + 1j * self.b / 2) * self.s_n / self.s_n_sys

    def get_admittance_off_diagonal(self):
        """
        Calculates the off-diagonal admittance value of the transformer.

        Returns:
            torch.Tensor: Off-diagonal admittance of the transformer.
        """
        return -1 / (self.r + 1j * self.x) * self.s_n / self.s_n_sys

    def enable_parallel_simulation(self, parallel_sims):
        """
        Enables parallel simulations for the transformer.

        Args:
            parallel_sims (int): Number of parallel simulations.
        """
        self.from_bus_id = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.from_bus_id
        self.to_bus_id = torch.ones((parallel_sims, 1), dtype=torch.int32) * self.to_bus_id
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

    def __init__(self, s_n_sys,
                 param_dict=None,
                 name=None,
                 bus=None,
                 p=None,
                 q=None,
                 model=None):
        """
        Initializes the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the load.
            name (str, optional): The name of the load.
            bus (str, optional): The name of the bus where the load is connected.
            p (float, optional): The active power demand of the load in MW.
            q (float, optional): The reactive power demand of the load in MVAR.
            model (str, optional): The model of the load. Currently only 'Z' is supported.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'bus': bus,
                'P': p,
                'Q': q,
                'model': model,
            }
        self.name = param_dict['name']
        self.bus = param_dict['bus']
        self.p_soll_mw = param_dict['P']
        self.q_soll_mvar = param_dict['Q']
        self.model = param_dict['model']

        if not self.model == 'Z':
            raise ValueError('Only Z model is supported for loads')

        self.s_n_sys = s_n_sys

        self.y_load = None

    def get_lf_power(self):
        """
        Calculates and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power (sum of active and reactive power)
        of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """

        return -(self.p_soll_mw + 1j * self.q_soll_mvar) / self.s_n_sys

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

    # noinspection PyUnusedLocal
    def initialize(self, s_calc, v_bb):
        """
        Initializes the Load by calculating its admittance.

        This method computes the Load's admittance based on the calculated complex power and the voltage base values.

        Args:
            s_calc (float): The calculated complex power.
            v_bb (torch.Tensor): Busbar voltage value.
        """
        s_load = (self.p_soll_mw + 1j * self.q_soll_mvar) / self.s_n_sys
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

    def __init__(self, s_n_sys,
                 param_dict=None,
                 name=None,
                 bus=None,
                 v_n=None,
                 q=None,
                 model=None):
        """
        Initializes the Load object with specified active and reactive power demands.

        Args:
            param_dict (dict, optional): Dictionary of parameters for the load.
            name (str, optional): The name of the load.
            bus (str, optional): The name of the bus where the load is connected.
            v_n (float, optional): The nominal voltage of the load in kV.
            q (float, optional): The reactive power demand of the load in MVAR.
            model (str, optional): The model of the load. Currently only 'Z' is supported.
        """
        if param_dict is None:
            param_dict = {
                'name': name,
                'bus': bus,
                'V_n': v_n,
                'Q': q,
                'model': model,
            }
        self.name = param_dict['name']
        self.bus = param_dict['bus']
        self.v_n = param_dict['V_n']
        self.q_soll_mvar = param_dict['Q']
        self.model = param_dict['model']

        if not self.model == 'Z':
            raise ValueError('Only Z model is supported for shunts')

        self.s_n_sys = s_n_sys

        s_shunt = torch.tensor(-1j * self.q_soll_mvar / self.s_n_sys)
        z = torch.conj(1 / s_shunt)
        self.y_shunt = 1 / z

    def get_lf_power(self):
        """
        Calculates and returns the load flow power of the Load.

        This method computes the load flow power by dividing the complex power (sum of active and reactive power)
        of the load by the system's base power.

        Returns:
            complex: The calculated load flow power.
        """

        return torch.zeros_like(self.q_soll_mvar)

    # noinspection PyUnusedLocal
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
