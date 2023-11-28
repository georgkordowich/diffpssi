from itertools import count
import time
from power_sim_lib.models.backend import *


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

        Parameters:
            ps_sim (PowerSystemSimulation): The power system simulation object to be integrated.
        """
        # calculate bus voltages
        voltages = torch.matmul(torch.linalg.inv(ps_sim.admittance_matrix(dynamic=True)), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])

            for model in bus.models:
                model_id = id(model)  # Unique identifier for each model
                dxdt_0 = model.differential()
                # Use previously stored x_1 if available, else use current state vector
                x_0 = self.x_0_store.get(model_id, model.get_state_vector())
                x_1 = x_0 + dxdt_0 * ps_sim.time_step
                model.set_state_vector(x_1)
                # Store x_1 for next step
                self.x_0_store[model_id] = x_1


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

        Parameters:
            ps_sim (PowerSystemSimulation): The power system simulation object to be integrated.
        """
        # calculate bus voltages
        voltages = torch.matmul(torch.linalg.inv(ps_sim.admittance_matrix(dynamic=True)), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                model_id = id(model)  # Unique identifier for each model
                dxdt_0_guess = model.differential()
                # Use previously stored x_1 if available, else use current state vector
                x_0 = self.x_0_store.get(model_id, model.get_state_vector())

                # calculate guess for x_1
                x_1_guess = x_0 + dxdt_0_guess * ps_sim.time_step
                model.set_state_vector(x_1_guess)
                self.dxdt_0_store[model_id] = dxdt_0_guess

        voltages = torch.matmul(torch.linalg.inv(ps_sim.admittance_matrix(dynamic=True)), ps_sim.current_injections())
        for i, bus in enumerate(ps_sim.busses):
            bus.update_voltages(voltages[:, i])
            for model in bus.models:
                model_id = id(model)
                dxdt_1_guess = model.differential()
                dxdt_est = (self.dxdt_0_store[model_id] + dxdt_1_guess) / 2
                x_0 = self.x_0_store.get(model_id, model.get_state_vector())
                x_1 = x_0 + dxdt_est * ps_sim.time_step
                # Store x_1 for next step
                model.set_state_vector(x_1)
                self.x_0_store[model_id] = x_1


solver_dict = {
    'euler': Euler,
    'heun': Heun,
}