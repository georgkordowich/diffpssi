import numpy as np
from tqdm import tqdm

from power_sim_lib.load_flow import do_load_flow
from power_sim_lib.models.synchronous_machine import SynchMachine
from power_sim_lib.models.static_models import *
from power_sim_lib.models.backend import *
from power_sim_lib.solvers import solver_dict

class PowerSystemSimulation(object):
    """
    Class representing a power system simulation.

    Attributes:
        time (numpy.ndarray): An array of time steps for the simulation.
        time_step (float): The time step interval.
        busses (list): List of bus objects in the system.
        non_slack_busses (list): List of non-slack buses in the system (currently unused).
        bus_names (dict): Dictionary mapping bus names to their indices.
        lines (list): List of line objects in the system.
        trafos (list): List of transformer objects in the system.
        fn (float): System frequency in Hz.
        base_mva (float): Base power in MVA.
        base_voltage (float): Base voltage.
        dynamic_y_matrix (torch.Tensor): Admittance matrix for dynamic analysis.
        static_y_matrix (torch.Tensor): Admittance matrix for static analysis.
        sc_event (ScEvent): Short circuit event object (if any).
        parallel_sims (int): Number of parallel simulations to run.
        record_func (function): Function to record simulation data.
        verbose (bool): Flag for verbose output.
        solver (Solver): Solver object for the simulation.

    Methods:
        add_bus: Adds a bus to the simulation.
        add_generator: Adds a generator to a specified bus.
        add_load: Adds a load to a specified bus.
        add_line: Adds a transmission line between two buses.
        add_trafo: Adds a transformer between two buses.
        admittance_matrix: Computes and returns the admittance matrix.
        current_injections: Computes current injections at each bus.
        initialize: Initializes the simulation state.
        add_sc_event: Adds a short circuit event to the simulation.
        set_record_function: Sets a custom function to record simulation data.
        reset: Resets the simulation to its initial state.
        run: Runs the simulation.
    """
    def __init__(self,
                 fn,
                 base_mva,
                 base_voltage,
                 time_step=0.005,
                 sim_time=5,
                 parallel_sims=1,
                 verbose=True,
                 solver='euler'):
        """
        Initializes the PowerSystemSimulation object.

        Parameters:
            fn (float): System frequency in Hz.
            base_mva (float): Base power in MVA.
            base_voltage (float): Base voltage.
            time_step (float): Time step for the simulation.
            sim_time (float): Total simulation time.
            parallel_sims (int): Number of parallel simulations.
            verbose (bool): Flag for verbose output.
            solver (str): Name of the solver to use.
        """
        self.time = np.arange(0, sim_time, time_step)
        self.time_step = time_step

        self.busses = []
        self.non_slack_busses = []
        self.bus_names = {}
        self.lines = []
        self.trafos = []

        self.fn = fn
        self.base_mva = base_mva
        self.base_voltage = base_voltage

        self.dynamic_y_matrix = None
        self.static_y_matrix = None

        self.sc_event = None
        self.parallel_sims = parallel_sims
        self.record_func = None
        self.verbose = verbose

        self.solver = solver_dict[solver]()

    def add_bus(self, name, v_n, lf_type='PV'):
        """
        Adds a bus to the system.

        Parameters:
            name (str): Name of the bus.
            v_n (float): Nominal voltage of the bus.
            lf_type (str): Load flow type of the bus, default is 'PV'.
        """
        # Assign an index to the bus and add it to the list of buses
        self.bus_names[name] = len(self.busses)
        self.busses.append(Bus(name, lf_type, v_n, self.parallel_sims))

    def add_generator(self, bus, param_dict):
        """
        Adds a generator to a specified bus in the system.

        Parameters:
            bus (str): The name of the bus to which the generator will be added.
            param_dict (dict): Dictionary containing the parameters of the generator.
        """
        self.busses[self.bus_names[bus]].add_model(SynchMachine(param_dict, parallel_sims=self.parallel_sims))
        # update the system variables of generator
        self.busses[self.bus_names[bus]].models[-1].set_sys_vars(self.fn, self.base_mva, self.base_voltage)
        # fit the voltage of the bus to the generator
        self.busses[self.bus_names[bus]].update_voltages(self.busses[self.bus_names[bus]].models[-1].v_soll)

    def add_load(self, bus, p_soll_mw, q_soll_mvar):
        """
        Adds a load to a specified bus in the system.

        Parameters:
            bus (str): The name of the bus to which the load will be added.
            p_soll_mw (float): The desired active power (MW) of the load.
            q_soll_mvar (float): The desired reactive power (MVAR) of the load.
        """
        self.busses[self.bus_names[bus]].add_model(Load(p_soll_mw=p_soll_mw, q_soll_mvar=q_soll_mvar, sys_s_n=self.base_mva, parallel_sims=self.parallel_sims))

    def add_line(self, from_bus, to_bus, r, x, b, length, unit):
        """
        Adds a transmission line between two buses in the system.

        Parameters:
            from_bus (str): The name of the starting bus for the line.
            to_bus (str): The name of the ending bus for the line.
            r (float): Resistance of the line.
            x (float): Reactance of the line.
            b (float): Susceptance of the line.
            length (float): Length of the line.
            unit (str): Unit of the line length (e.g., km, miles).
        """
        self.lines.append(Line(self.bus_names[from_bus], self.bus_names[to_bus], r, x, b, length, unit, self.base_mva,
                               self.base_voltage, parallel_sims=self.parallel_sims))

    def add_trafo(self, from_bus, to_bus, s_n, r, x, v_n_from, v_n_to):
        """
        Adds a transformer between two buses in the system.

        Parameters:
            from_bus (str): The name of the primary side bus of the transformer.
            to_bus (str): The name of the secondary side bus of the transformer.
            s_n (float): Rated power of the transformer.
            r (float): Resistance of the transformer.
            x (float): Reactance of the transformer.
            v_n_from (float): Nominal voltage of the primary side.
            v_n_to (float): Nominal voltage of the secondary side.
        """
        self.trafos.append(Transfomer(self.bus_names[from_bus], self.bus_names[to_bus], s_n, r, x,
                                      v_n_from, v_n_to, self.base_mva,
                                      parallel_sims=self.parallel_sims))

    def admittance_matrix(self, dynamic):
        """
        Computes and returns the admittance matrix for the system, either dynamic or static.

        Parameters:
            dynamic (bool): Flag to choose between dynamic (True) or static (False) admittance matrix.

        Returns:
            torch.Tensor: The computed admittance matrix.
        """
        if dynamic and self.dynamic_y_matrix is not None:
            return self.dynamic_y_matrix
        elif not dynamic and self.static_y_matrix is not None:
            return self.static_y_matrix
        elif dynamic and self.dynamic_y_matrix is None:
            self.dynamic_y_matrix = torch.zeros((self.parallel_sims, len(self.busses), len(self.busses)), dtype=torch.complex128)
            for line in self.lines:
                self.dynamic_y_matrix[:, line.from_bus, line.to_bus] = line.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, line.to_bus, line.from_bus] = line.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, line.from_bus, line.from_bus] += line.get_admittance_diagonal()
                self.dynamic_y_matrix[:, line.to_bus, line.to_bus] += line.get_admittance_diagonal()

            for transformer in self.trafos:
                self.dynamic_y_matrix[:, transformer.from_bus, transformer.to_bus] = transformer.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, transformer.to_bus, transformer.from_bus] = transformer.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, transformer.from_bus, transformer.from_bus] += transformer.get_admittance_diagonal()
                self.dynamic_y_matrix[:, transformer.to_bus, transformer.to_bus] += transformer.get_admittance_diagonal()

            for i, bus in enumerate(self.busses):
                for model in bus.models:
                    self.dynamic_y_matrix[:, i, i] += model.get_admittance(dynamic).squeeze()
            return self.dynamic_y_matrix
        else:
            self.static_y_matrix = torch.zeros((self.parallel_sims, len(self.busses), len(self.busses)), dtype=torch.complex128)
            for line in self.lines:
                self.static_y_matrix[:, line.from_bus, line.to_bus] = line.get_admittance_off_diagonal()
                self.static_y_matrix[:, line.to_bus, line.from_bus] = line.get_admittance_off_diagonal()
                self.static_y_matrix[:, line.from_bus, line.from_bus] += line.get_admittance_diagonal()
                self.static_y_matrix[:, line.to_bus, line.to_bus] += line.get_admittance_diagonal()

            for transformer in self.trafos:
                self.static_y_matrix[:, transformer.from_bus, transformer.to_bus] = transformer.get_admittance_off_diagonal()
                self.static_y_matrix[:, transformer.to_bus, transformer.from_bus] = transformer.get_admittance_off_diagonal()
                self.static_y_matrix[:, transformer.from_bus, transformer.from_bus] += transformer.get_admittance_diagonal()
                self.static_y_matrix[:, transformer.to_bus, transformer.to_bus] += transformer.get_admittance_diagonal()

            for i, bus in enumerate(self.busses):
                for model in bus.models:
                    self.static_y_matrix[:, i, i] += model.get_admittance(dynamic).squeeze()
            return self.static_y_matrix

    def current_injections(self):
        """
        Computes the current injections at each bus in the system.

        Returns:
            torch.Tensor: A tensor representing current injections at each bus.
        """

        return torch.stack([bus.get_current_injections() for bus in self.busses], axis=1)

    def initialize(self):
        """
        Initializes the simulation state by setting up initial conditions and computing initial values.
        """
        power_inj = do_load_flow(self)
        for i, bus in enumerate(self.busses):
            for model in bus.models:
                model.initialize(power_inj[:, i], bus.voltage)
        # calculate bus voltages
        voltages = torch.matmul(torch.linalg.inv(self.admittance_matrix(dynamic=True)), self.current_injections())

        for i, bus in enumerate(self.busses):
            bus.update_voltages(voltages[:, i])

    def add_sc_event(self, start_time, end_time, bus):
        """
        Adds a short circuit event to the simulation.

        Parameters:
            start_time (float): The start time of the short circuit event.
            end_time (float): The end time of the short circuit event.
            bus (str): The name of the bus where the short circuit occurs.
        """
        bus_idx = self.bus_names[bus]
        self.sc_event = ScEvent(start_time, end_time, bus_idx)

    def set_record_function(self, record_func):
        """
        Sets a custom function to record simulation data.

        Parameters:
            record_func (function): A function that defines how simulation data is recorded.
        """
        self.record_func = record_func

    def reset(self):
        """
        Resets the simulation to its initial state. This includes resetting all model states and matrices.
        """
        # reset all model states
        for bus in self.busses:
            bus.reset()
            for model in bus.models:
                model.reset()
            bus.update_voltages(bus.models[-1].v_soll)

        self.static_y_matrix = None
        self.dynamic_y_matrix = None

    def run(self):
        """
        Runs the simulation. It initializes the system, runs through the simulation time steps, and records the system state.

        Returns:
            tuple: A tuple containing the simulation time steps and a tensor of recorded data.
        """
        self.initialize()

        recorder_list = []
        # copy the tensor of y_matrix
        if BACKEND == 'numpy':
            original_y_matrix = self.dynamic_y_matrix.copy()
        elif BACKEND == 'torch':
            original_y_matrix = self.dynamic_y_matrix.clone()
        else:
            raise ValueError('Backend not recognized')

        if self.verbose:
            iterator = tqdm(self.time)
        else:
            iterator = self.time

        for t in iterator:
            if (self.sc_event is not None) and self.sc_event.is_active(t):
                # if the sc_event is active set the y_matrix to a matrix with a short circuit at the specified bus
                self.dynamic_y_matrix[:, self.sc_event.bus, self.sc_event.bus] = 1e6
            else:
                self.dynamic_y_matrix[:, :, :] = original_y_matrix

            # do a step with the solver
            self.solver.step(self)

            # Record the state of the system
            try:
                recorder_list.append(torch.stack(self.record_func(self)))
            except TypeError:
                print('No record function specified')

        # Format shall be [batch, timestep, value]
        return_tensor = torch.swapaxes(torch.stack(recorder_list, axis=1), 0, 2).squeeze(-1)
        return self.time, return_tensor
