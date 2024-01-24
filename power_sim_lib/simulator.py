"""
The main simulation class. This class represents a power system simulation. It contains all the necessary information
about the system, such as the buses, lines, transformers, etc. It also contains the admittance matrix, which is
computed based on the system configuration. The simulation can be run by calling the run() method.
"""
import time
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
        bus_idxs (dict): Dictionary mapping bus names to their indices.
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
                 time_step,
                 sim_time,
                 parallel_sims,
                 solver,
                 grid_data,
                 verbose=True,
                 ):
        """
        Initializes the PowerSystemSimulation object.

        Args:
            time_step (float): Time step for the simulation.
            sim_time (float): Total simulation time.
            parallel_sims (int): Number of parallel simulations.
            verbose (bool): Flag for verbose output.
            solver (str): Name of the solver to use.
        """
        # Add timestep in the end because the first step does not have a value
        self.time = np.arange(0, sim_time, time_step)
        self.time_step = time_step

        self.busses = []
        self.non_slack_busses = []
        self.bus_idxs = {}
        self.lines = []
        self.trafos = []

        self.dynamic_y_matrix = None
        self.static_y_matrix = None

        self.sc_event = None
        self.parallel_sims = parallel_sims
        self.record_func = None
        self.verbose = verbose

        self.solver = solver_dict[solver]()

        self.fn = grid_data['f']
        self.base_mva = grid_data['base_mva']
        self.base_voltage = grid_data['base_voltage']
        self.create_grid(grid_data)

    def get_generator_by_name(self, name):
        """
        Returns a generator object by its name.
        Args:
            name:

        Returns:

        """
        for bus in self.busses:
            for model in bus.models:
                if model.name == name:
                    return model
        return None

    def create_grid(self, grid_data):
        """
        Creates the grid based on the provided grid data.

        Args:
            grid_data (dict): Dictionary containing the grid data.
        """
        transformed_data = {}
        # first transform the data to a more convenient format
        for key, value in grid_data.items():
            # Check if the value is a list of lists
            if isinstance(value, list) and all(isinstance(item, list) for item in value):
                # Use the first sublist as keys, and transform the remaining sublists into dictionaries
                keys = value[0]
                transformed_data[key] = [dict(zip(keys, v)) for v in value[1:]]
            elif isinstance(value, dict):
                # If the value is a dictionary, apply the transformation to each key within the dictionary
                transformed_data[key] = {sub_key: [dict(zip(value[sub_key][0], v)) for v in value[sub_key][1:]]
                                         for sub_key in value}
            else:
                # Copy the value as is
                transformed_data[key] = value
        grid_data = transformed_data

        for bus in grid_data['busses']:
            self.add_bus(bus)

        generators = grid_data.get('generators', [])
        if isinstance(generators, dict):
            for model in generators['GEN']:
                self.add_generator(model)

        for load in grid_data.get('loads', []):
            self.add_load(load)

        for shunt in grid_data.get('shunts', []):
            self.add_shunt(shunt)

        for line in grid_data.get('lines', []):
            self.add_line(line)

        for transformer in grid_data.get('transformers', []):
            self.add_transformer(transformer)

        exciters = grid_data.get('avr', [])
        if isinstance(exciters, dict):
            for model in exciters['SEXS']:
                self.get_generator_by_name(model['gen']).add_exciter(model, self.parallel_sims)

        governors = grid_data.get('gov', [])
        if isinstance(governors, dict):
            for model in governors['TGOV1']:
                self.get_generator_by_name(model['gen']).add_governor(model, self.parallel_sims)

        psss = grid_data.get('pss', [])
        if isinstance(psss, dict):
            for model in psss['STAB1']:
                self.get_generator_by_name(model['gen']).add_pss(model, self.parallel_sims)

        self.busses[self.bus_idxs[grid_data['slack_bus']]].lf_type = 'SL'

    def add_bus(self, param_dict):
        """
        Adds a bus to the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the bus.
        """
        # Assign an index to the bus and add it to the list of buses
        self.bus_idxs[param_dict['name']] = len(self.busses)
        self.busses.append(Bus(param_dict, self.parallel_sims))

    def add_generator(self, param_dict):
        """
        Adds a generator to a specified bus in the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the generator.
        """
        bus = self.busses[self.bus_idxs[param_dict['bus']]]
        bus.add_model(SynchMachine(param_dict, self.fn, self.base_mva, bus.v_n, self.parallel_sims))
        # fit the voltage of the bus to the generator
        bus.update_voltages(bus.models[-1].v_soll)
        bus.lf_type = 'PV'

    def add_load(self, param_dict):
        """
        Adds a load to a specified bus in the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the load.
        """
        bus = self.busses[self.bus_idxs[param_dict['bus']]]
        bus.add_model(Load(param_dict, self.base_mva, bus.v_n, self.parallel_sims))

    def add_shunt(self, param_dict):
        """
        Adds a load to a specified bus in the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the shunt element.
        """
        bus = self.busses[self.bus_idxs[param_dict['bus']]]
        bus.add_model(Shunt(param_dict, self.base_mva, bus.v_n, self.parallel_sims))

    def add_line(self, param_dict):
        """
        Adds a transmission line between two buses in the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the line.
        """
        bus_from = self.bus_idxs[param_dict['from_bus']]
        bus_to = self.bus_idxs[param_dict['to_bus']]
        self.lines.append(Line(param_dict,
                               bus_from,
                               bus_to,
                               self.base_mva,
                               self.base_voltage,
                               parallel_sims=self.parallel_sims))

    def add_transformer(self, param_dict):
        """
        Adds a transformer between two buses in the system.

        Args:
            param_dict (dict): Dictionary containing the parameters of the transformer.
        """
        bus_from = self.bus_idxs[param_dict['from_bus']]
        bus_to = self.bus_idxs[param_dict['to_bus']]
        self.trafos.append(Transfomer(param_dict,
                                      bus_from,
                                      bus_to,
                                      s_n_sys=self.base_mva,
                                      parallel_sims=self.parallel_sims))

    def admittance_matrix(self, dynamic):
        """
        Computes and returns the admittance matrix for the system, either dynamic or static.

        Args:
            dynamic (bool): Flag to choose between dynamic (True) or static (False) admittance matrix.

        Returns:
            torch.Tensor: The computed admittance matrix.
        """
        if dynamic and self.dynamic_y_matrix is not None:
            # get the previously computed dynamic y_matrix
            return self.dynamic_y_matrix
        elif not dynamic and self.static_y_matrix is not None:
            # get the previously computed static y_matrix
            return self.static_y_matrix
        elif dynamic and self.dynamic_y_matrix is None:
            # reconstruct the dynamic y_matrix
            self.dynamic_y_matrix = torch.zeros((self.parallel_sims,
                                                 len(self.busses),
                                                 len(self.busses)),
                                                dtype=torch.complex128)
            for line in self.lines:
                self.dynamic_y_matrix[:, line.from_bus, line.to_bus] += line.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, line.to_bus, line.from_bus] += line.get_admittance_off_diagonal()
                self.dynamic_y_matrix[:, line.from_bus, line.from_bus] += line.get_admittance_diagonal()
                self.dynamic_y_matrix[:, line.to_bus, line.to_bus] += line.get_admittance_diagonal()

            for transformer in self.trafos:
                self.dynamic_y_matrix[:, transformer.from_bus, transformer.to_bus] += (
                    transformer.get_admittance_off_diagonal())
                self.dynamic_y_matrix[:, transformer.to_bus, transformer.from_bus] += (
                    transformer.get_admittance_off_diagonal())
                self.dynamic_y_matrix[:, transformer.from_bus, transformer.from_bus] += (
                    transformer.get_admittance_diagonal())
                self.dynamic_y_matrix[:, transformer.to_bus, transformer.to_bus] += (
                    transformer.get_admittance_diagonal())

            for i, bus in enumerate(self.busses):
                for model in bus.models:
                    self.dynamic_y_matrix[:, i, i] += model.get_admittance(dynamic).squeeze()
            return self.dynamic_y_matrix
        else:
            # reconstruct the static y_matrix
            self.static_y_matrix = torch.zeros((self.parallel_sims,
                                                len(self.busses),
                                                len(self.busses)),
                                               dtype=torch.complex128)
            for line in self.lines:
                self.static_y_matrix[:, line.from_bus, line.to_bus] += line.get_admittance_off_diagonal()
                self.static_y_matrix[:, line.to_bus, line.from_bus] += line.get_admittance_off_diagonal()
                self.static_y_matrix[:, line.from_bus, line.from_bus] += line.get_admittance_diagonal()
                self.static_y_matrix[:, line.to_bus, line.to_bus] += line.get_admittance_diagonal()

            for transformer in self.trafos:
                self.static_y_matrix[:, transformer.from_bus, transformer.to_bus] += (
                    transformer.get_admittance_off_diagonal())
                self.static_y_matrix[:, transformer.to_bus, transformer.from_bus] += (
                    transformer.get_admittance_off_diagonal())
                self.static_y_matrix[:, transformer.from_bus, transformer.from_bus] += (
                    transformer.get_admittance_diagonal())
                self.static_y_matrix[:, transformer.to_bus, transformer.to_bus] += (
                    transformer.get_admittance_diagonal())

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

        Args:
            start_time (float): The start time of the short circuit event.
            end_time (float): The end time of the short circuit event.
            bus (str): The name of the bus where the short circuit occurs.
        """
        bus_idx = self.bus_idxs[bus]
        self.sc_event = ScEvent(start_time, end_time, bus_idx)

    def set_record_function(self, record_func):
        """
        Sets a custom function to record simulation data.

        Args:
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
            pass

        self.solver.reset()
        self.static_y_matrix = None
        self.dynamic_y_matrix = None

    def run(self):
        """
        Runs the simulation. It initializes the system, runs through the simulation time steps, and records the system
        state.

        Returns:
            tuple: A tuple containing the simulation time steps and a tensor of recorded data.
        """
        self.initialize()

        start_time = time.time()

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
            self.solver.step()

            # Record the state of the system
            try:
                recorder_list.append(torch.stack(self.record_func(self)))
            except TypeError:
                print('No record function specified')

        # Format shall be [batch, timestep, value]
        return_tensor = torch.swapaxes(torch.stack(recorder_list, axis=1), 0, 2).squeeze(-1)

        if self.verbose:
            end_time = time.time()
            print('Dynamic simulation finished in {:.2f} seconds'.format(end_time - start_time))

        return self.time, return_tensor
