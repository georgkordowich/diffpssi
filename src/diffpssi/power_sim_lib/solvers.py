"""
File contains all the solvers for the power system simulation. The solvers are used to integrate the differential
equations of the power system simulation. More solvers can be added here
"""
from src.diffpssi.power_sim_lib.backend import *


class Euler(object):
    """
    Implements the Euler method for numerical integration in power system simulations.

    The Euler method is a first-order numerical procedure for solving ordinary differential equations (ODEs)
    with a given initial value. It is the most basic explicit method for numerical integration of ODEs
    and is the simplest Runge–Kutta method.

    Attributes:
        x_0_store (dict): A dictionary for storing the previous state vector of each model.
    """

    def __init__(self):
        """
        Initializes the Euler solver object.
        """
        self.x_0_store = {}

    def step(self, ps_sim):
        """
        Executes one step of the Euler integration method for the power system simulation.

        Args:
            ps_sim (PowerSystemSimulation): The power system simulation object to be integrated.
        """
        # calculate bus voltages
        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])

            for model in bus.diff_models:
                model_id = id(model)  # Unique identifier for each model
                dxdt_0 = model.differential()
                # Use previously stored x_1 if available, else use current state vector
                x_0 = self.x_0_store.get(model_id, model.get_state_vector())
                x_1 = x_0 + dxdt_0 * ps_sim.time_step
                model.set_state_vector(x_1)
                # Store x_1 for next step
                self.x_0_store[model_id] = x_1

    def reset(self):
        """
        Resets the Euler solver object.
        """
        self.x_0_store = {}


class Heun(object):
    """
    Implements the Heun method (or the improved Euler method) for numerical integration in power system simulations.

    The Heun method is a second-order numerical procedure for solving ordinary differential equations (ODEs).
    It's an explicit method that improves upon the basic Euler method by making a preliminary step with the Euler method
    and then an adjustment with a slope that is the average of the slopes at the two ends of the interval.

    Attributes:
        x_0_store (dict): A dictionary for storing the previous state vector of each model.
        dxdt_0_store (dict): A dictionary for storing the derivative of the state vector from the previous step.
    """

    def __init__(self):
        """
        Initializes the Heun solver object.
        """
        self.x_0_store = {}
        self.dxdt_0_store = {}

    def step(self, ps_sim):
        """
        Executes one step of the Heun integration method for the power system simulation.

        Args:
            ps_sim (PowerSystemSimulation): The power system simulation object to be integrated.
        """
        # calculate bus voltages
        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)  # Unique identifier for each model
                    dxdt_0_guess = model.differential()
                    # Use previously stored x_1 if available, else use current state vector
                    x_0 = self.x_0_store.get(model_id, model.get_state_vector())

                    # calculate guess for x_1
                    x_1_guess = x_0 + dxdt_0_guess * ps_sim.time_step
                    model.set_state_vector(x_1_guess)
                    self.dxdt_0_store[model_id] = dxdt_0_guess
                    self.x_0_store[model_id] = x_0
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass

        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)
                    dxdt_1_guess = model.differential()
                    dxdt_est = (self.dxdt_0_store[model_id] + dxdt_1_guess) / 2
                    x_0 = self.x_0_store.get(model_id)
                    x_1 = x_0 + dxdt_est * ps_sim.time_step
                    # Store x_1 for next step
                    model.set_state_vector(x_1)
                    self.x_0_store[model_id] = x_1
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass

    def reset(self):
        """
        Resets the Heun solver object.
        """
        self.x_0_store = {}
        self.dxdt_0_store = {}


class RK4(object):
    """
    Implements the Runge-Kutta 4 method for numerical integration in power system simulations.

    Attributes:
        x_0_store (dict): A dictionary for storing the previous state vector of each model.
        k1_store (dict): A dictionary for storing the k1 values of each model.
        k2_store (dict): A dictionary for storing the k2 values of each model.
        k3_store (dict): A dictionary for storing the k3 values of each model.
    """
    def __init__(self):
        self.x_0_store = {}
        self.k1_store = {}
        self.k2_store = {}
        self.k3_store = {}

    def step(self, ps_sim):
        """
        Executes one step of the Runge-Kutta 4 integration method for the power system simulation.
        Args:
            ps_sim: The power system simulation object to be integrated.
        """
        # calculate bus voltages
        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        # calc k1
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)  # Unique identifier for each model
                    k1 = model.differential()
                    # Use previously stored x_1 if available, else use current state vector
                    x_0 = self.x_0_store.get(model_id, model.get_state_vector())

                    # calculate guess for x_1
                    x_k2 = x_0 + k1 * ps_sim.time_step / 2
                    model.set_state_vector(x_k2)
                    self.k1_store[model_id] = k1

                    self.x_0_store[model_id] = x_0
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass

        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        # calc k2
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)
                    k2 = model.differential()
                    x_0 = self.x_0_store.get(model_id)
                    x_k3 = x_0 + k2 * ps_sim.time_step / 2
                    model.set_state_vector(x_k3)
                    self.k2_store[model_id] = k2
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass
        # calc k3
        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)
                    k3 = model.differential()
                    x_0 = self.x_0_store.get(model_id)
                    x_k4 = x_0 + k3 * ps_sim.time_step
                    model.set_state_vector(x_k4)
                    self.k3_store[model_id] = k3
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass

        voltages = torch.matmul(ps_sim.inverse_dyn_admittance_matrix(), ps_sim.current_injections())
        # calc k4
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                try:
                    model_id = id(model)
                    k4 = model.differential()
                    x_0 = self.x_0_store.get(model_id)
                    x_1 = x_0 + (self.k1_store[model_id] + 2 * self.k2_store[model_id] + 2 * self.k3_store[
                        model_id] + k4) * ps_sim.time_step / 6
                    model.set_state_vector(x_1)
                    self.x_0_store[model_id] = x_1
                except AttributeError:
                    # This happens for models that do not have a differential function
                    pass

    def reset(self):
        """
        Resets the RK4 solver object, so a new simulation can be started.
        """
        self.x_0_store = {}


solver_dict = {
    'euler': Euler,
    'heun': Heun,
    'rk4': RK4,
}
