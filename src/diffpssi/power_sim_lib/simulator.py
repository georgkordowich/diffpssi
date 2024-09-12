"""
The main simulation class. This class represents a power system simulation. It contains all the necessary information
about the system, such as the buses, lines, transformers, etc. It also contains the admittance matrix, which is
computed based on the system configuration. The simulation can be run by calling the run() method.
"""
import os
import time
import numpy as np
from tqdm import tqdm

from src.diffpssi.power_sim_lib.load_flow import do_load_flow
from src.diffpssi.power_sim_lib.models.synchronous_machine import SynchMachine
from src.diffpssi.power_sim_lib.models.static_models import *
from src.diffpssi.power_sim_lib.backend import *
from src.diffpssi.power_sim_lib.solvers import solver_dict
from src.diffpssi.power_sim_lib.models.governors import TGOV1
from src.diffpssi.power_sim_lib.models.stabilizers import STAB1
from src.diffpssi.power_sim_lib.models.exciters import SEXS


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
        sc_events (ScEvent): Short circuit event object (if any).
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
                 grid_data=None,
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

        self.inverse_dynamic_y_matrix = None
        self.static_y_matrix = None

        self.sc_events = []
        self.param_events = []
        self.parallel_sims = parallel_sims
        self.record_func = None
        self.verbose = verbose

        if os.environ.get('DIFFPSSI_FORCE_INTEGRATOR') is not None:
            # this should only be used for integration tests
            self.solver = solver_dict[os.environ.get('DIFFPSSI_FORCE_INTEGRATOR')]()
            print('WARNING: FORCING THE USE OF THE {} INTEGRATOR. '
                  'THIS SHOULD ONLY HAPPEN FOR UNITTESTS'.format(os.environ.get('DIFFPSSI_FORCE_INTEGRATOR')))
        else:
            self.solver = solver_dict[solver]()

        self.base_voltage = None
        self.base_mva = None
        self.fn = None

        if grid_data is not None:
            self.create_grid(grid_data)

        self.backend = BACKEND

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
        self.fn = grid_data['f']
        self.base_mva = grid_data['base_mva']
        self.base_voltage = grid_data['base_voltage']

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

        for bus_dict in grid_data['busses']:
            bus_model = Bus(param_dict=bus_dict)
            bus_model.enable_parallel_simulation(self.parallel_sims)
            self.add_bus(bus_model)

        generators = grid_data.get('generators', [])
        if isinstance(generators, dict):
            for gen_dict in generators['GEN']:
                generator_model = SynchMachine(param_dict=gen_dict,
                                               s_n_sys=self.base_mva,
                                               v_n_sys=self.base_voltage,
                                               f_n_sys=self.fn)
                self.add_generator(generator_model)

        for load_dict in grid_data.get('loads', []):
            load_model = Load(param_dict=load_dict,
                              s_n_sys=self.base_mva)
            self.add_load(load_model)

        for shunt_dict in grid_data.get('shunts', []):
            shunt_model = Shunt(param_dict=shunt_dict,
                                s_n_sys=self.base_mva)
            self.add_shunt(shunt_model)

        for line_dict in grid_data.get('lines', []):
            line_model = Line(param_dict=line_dict,
                              s_n_sys=self.base_mva,
                              v_n_sys=self.base_voltage)
            self.add_line(line_model)

        for transformer_dict in grid_data.get('transformers', []):
            transformer_model = Transformer(param_dict=transformer_dict,
                                            s_n_sys=self.base_mva)
            self.add_transformer(transformer_model)

        exciters = grid_data.get('avr', [])
        if isinstance(exciters, dict):
            for sexs_dict in exciters['SEXS']:
                exciter_model = SEXS(param_dict=sexs_dict)
                self.add_exciter(exciter_model)

        governors = grid_data.get('gov', [])
        if isinstance(governors, dict):
            for gov_dict in governors['TGOV1']:
                gov_model = TGOV1(param_dict=gov_dict)
                self.add_governor(gov_model)

        psss = grid_data.get('pss', [])
        if isinstance(psss, dict):
            for pss_dict in psss['STAB1']:
                pss_model = STAB1(param_dict=pss_dict)
                self.add_pss(pss_model)

        self.set_slack_bus(grid_data['slack_bus'])

    def set_slack_bus(self, slack_bus):
        """
        Sets the slack bus of the system.

        Args:
            slack_bus (str): The name of the slack bus.
        """
        slack_bus_idx = self.bus_idxs[slack_bus]
        self.busses[slack_bus_idx].lf_type = 'SL'

    def add_bus(self, bus_model):
        """
        Adds a bus to the system.

        Args:
            bus_model (Bus): A bus model to add to the grid.
        """
        # Assign an index to the bus and add it to the list of buses
        self.bus_idxs[bus_model.name] = len(self.busses)
        bus_model.enable_parallel_simulation(self.parallel_sims)
        self.busses.append(bus_model)

    def add_generator(self, generator_model):
        """
        Adds a generator to a specified bus in the system.

        Args:
            generator_model (SynchMachine): A generator model to add to the grid.
        """
        bus = self.busses[self.bus_idxs[generator_model.bus]]
        generator_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(generator_model)
        # fit the voltage of the bus to the generator
        bus.update_voltages(bus.models[-1].v_soll)
        bus.lf_type = 'PV'

    def add_inverter(self, inverter_model):
        bus = self.busses[self.bus_idxs[inverter_model.bus]]
        inverter_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(inverter_model)

        bus.lf_type = 'PQ'

    def add_load(self, load_model):
        """
        Adds a load to a specified bus in the system.

        Args:
            load_model (Load): A load model to add to the grid.
        """
        bus = self.busses[self.bus_idxs[load_model.bus]]
        load_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(load_model)

    def add_shunt(self, shunt_model):
        """
        Adds a shunt to a specified bus in the system.

        Args:
            shunt_model (Shunt): A shunt model to add to the grid.
        """
        bus = self.busses[self.bus_idxs[shunt_model.bus]]
        shunt_model.enable_parallel_simulation(self.parallel_sims)
        bus.add_model(shunt_model)

    def add_line(self, line_model):
        """
        Adds a transmission line between two buses in the system.

        Args:
            line_model (Line): A line model to add to the grid.
        """
        bus_from = self.bus_idxs[line_model.from_bus_name]
        bus_to = self.bus_idxs[line_model.to_bus_name]

        line_model.from_bus_id = bus_from
        line_model.to_bus_id = bus_to

        line_model.enable_parallel_simulation(self.parallel_sims)

        self.lines.append(line_model)

    def add_transformer(self, transformer_model):
        """
        Adds a transformer between two buses in the system.

        Args:
            transformer_model (Transformer): A transformer model to add to the grid.
        """
        bus_from = self.bus_idxs[transformer_model.from_bus_name]
        bus_to = self.bus_idxs[transformer_model.to_bus_name]

        transformer_model.from_bus_id = bus_from
        transformer_model.to_bus_id = bus_to

        transformer_model.enable_parallel_simulation(self.parallel_sims)

        self.trafos.append(transformer_model)

    def add_exciter(self, exciter_model):
        """
        Adds an exciter to a specified generator in the system.
        Different exciter models work, for example the SEXS.

        Args:
            exciter_model (object): An exciter model to add to the grid.
        """
        generator = self.get_generator_by_name(exciter_model.gen)
        exciter_model.v_setpoint = generator.v_soll
        exciter_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_exciter(exciter_model)

    def add_governor(self, governor_model):
        """
        Adds a governor to a specified generator in the system.
        Different governor models work, for example the TGOV1.

        Args:
            governor_model (object): A governor model to add to the grid.
        """
        generator = self.get_generator_by_name(governor_model.gen)
        governor_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_governor(governor_model)

    def add_pss(self, pss_model):
        """
        Adds a PSS to a specified generator in the system.
        Different PSS models work, for example the STAB1.

        Args:
            pss_model (object): A PSS model to add to the grid.
        """
        generator = self.get_generator_by_name(pss_model.gen)
        pss_model.enable_parallel_simulation(self.parallel_sims)

        generator.add_pss(pss_model)

    def inverse_dyn_admittance_matrix(self):
        """
        Computes and returns the inverse dynamic admittance matrix for the system, either dynamic or static.

        Returns:
            torch.Tensor: The computed admittance matrix.
        """
        if self.inverse_dynamic_y_matrix is not None:
            # get the previously computed dynamic y_matrix
            return self.inverse_dynamic_y_matrix
        else:
            # reconstruct the dynamic y_matrix
            dynamic_y_matrix = torch.zeros((self.parallel_sims,
                                            len(self.busses),
                                            len(self.busses)),
                                           dtype=torch.complex128)
            for line in self.lines:
                dynamic_y_matrix[:, line.from_bus_id, line.to_bus_id] += line.get_admittance_off_diagonal()
                dynamic_y_matrix[:, line.to_bus_id, line.from_bus_id] += line.get_admittance_off_diagonal()
                dynamic_y_matrix[:, line.from_bus_id, line.from_bus_id] += line.get_admittance_diagonal()
                dynamic_y_matrix[:, line.to_bus_id, line.to_bus_id] += line.get_admittance_diagonal()

            for transformer in self.trafos:
                dynamic_y_matrix[:, transformer.from_bus_id, transformer.to_bus_id] += (
                    transformer.get_admittance_off_diagonal())
                dynamic_y_matrix[:, transformer.to_bus_id, transformer.from_bus_id] += (
                    transformer.get_admittance_off_diagonal())
                dynamic_y_matrix[:, transformer.from_bus_id, transformer.from_bus_id] += (
                    transformer.get_admittance_diagonal())
                dynamic_y_matrix[:, transformer.to_bus_id, transformer.to_bus_id] += (
                    transformer.get_admittance_diagonal())

            for i, bus in enumerate(self.busses):
                for model in bus.models:
                    dynamic_y_matrix[:, i, i] += model.get_admittance(dyn=True).squeeze()

            self.inverse_dynamic_y_matrix = torch.linalg.inv(dynamic_y_matrix)
            return self.inverse_dynamic_y_matrix

    def lf_admittance_matrix(self):
        """
        Computes and returns the static admittance matrix for the system.

        Returns:
            torch.Tensor: The computed admittance matrix.
        """
        if self.static_y_matrix is not None:
            # get the previously computed static y_matrix
            return self.static_y_matrix
        # reconstruct the static y_matrix
        static_y_matrix = torch.zeros((self.parallel_sims,
                                       len(self.busses),
                                       len(self.busses)),
                                      dtype=torch.complex128)
        for line in self.lines:
            static_y_matrix[:, line.from_bus_id, line.to_bus_id] += line.get_admittance_off_diagonal()
            static_y_matrix[:, line.to_bus_id, line.from_bus_id] += line.get_admittance_off_diagonal()
            static_y_matrix[:, line.from_bus_id, line.from_bus_id] += line.get_admittance_diagonal()
            static_y_matrix[:, line.to_bus_id, line.to_bus_id] += line.get_admittance_diagonal()

        for transformer in self.trafos:
            static_y_matrix[:, transformer.from_bus_id, transformer.to_bus_id] += (
                transformer.get_admittance_off_diagonal())
            static_y_matrix[:, transformer.to_bus_id, transformer.from_bus_id] += (
                transformer.get_admittance_off_diagonal())
            static_y_matrix[:, transformer.from_bus_id, transformer.from_bus_id] += (
                transformer.get_admittance_diagonal())
            static_y_matrix[:, transformer.to_bus_id, transformer.to_bus_id] += (
                transformer.get_admittance_diagonal())

        for i, bus in enumerate(self.busses):
            for model in bus.models:
                static_y_matrix[:, i, i] += model.get_admittance(dyn=False).squeeze()

        self.static_y_matrix = static_y_matrix

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
        voltages = torch.matmul(self.inverse_dyn_admittance_matrix(), self.current_injections())

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
        self.sc_events.append(ScEvent(start_time, end_time, bus_idx))

    def add_param_event(self, timestep, model, parameter, new_val):
        """
        Adds a parameter event to the simulation.

        Args:
            timestep (float): The time step of the parameter event.
            model (object): The model object to change the parameter of.
            parameter (str): The name of the parameter to change.
            new_val (float): The new value of the parameter.
        """
        self.param_events.append(ParamEvent(timestep, model, parameter, new_val))

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

        self.solver.reset()
        self.static_y_matrix = None
        self.inverse_dynamic_y_matrix = None

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
            original_y_matrix = self.inverse_dynamic_y_matrix.copy()
        elif BACKEND == 'torch':
            original_y_matrix = self.inverse_dynamic_y_matrix.clone()
        else:
            raise ValueError('Backend not recognized')

        if self.verbose:
            iterator = tqdm(self.time)
        else:
            iterator = self.time

        for t in iterator:
            for sc_event in self.sc_events:
                if sc_event.is_active(t):
                    dynamic_y_matrix = torch.linalg.inv(self.inverse_dynamic_y_matrix)
                    dynamic_y_matrix[:, sc_event.bus, sc_event.bus] = 1e6
                    self.inverse_dynamic_y_matrix = torch.linalg.inv(dynamic_y_matrix)
                else:
                    # todo
                    self.inverse_dynamic_y_matrix = original_y_matrix

            for param_event in self.param_events:
                param_event.handle_event(t)

            # do a step with the solver
            self.solver.step(self)

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
