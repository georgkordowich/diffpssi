# DiffPSSi: A framework for differentiable power system simulations
## Quickstart
1. Install Package:
   ```
   pip install diffpssi
   ```
2. Copy-Paste example simulation from:
    ```
    examples/models/ibb_manual/ibb_sim_manual.py
    ```
3. Run the simulation:
    ```
    python ibb_sim_manual.py
    ```

## Overview
DiffPSSi contains a framework designed for simulating and optimizing the dynamic behavior 
of power systems. The core idea of the framework is to allow the use of automatic differentiation for dynamic 
power system simulations. Effectively, this allows for the calculation of gradients of all simulation parameters
with respect to a desired output of the simulation. This is useful for parameter optimization or identification
or concepts like known operator learning.

It includes detailed models of various power system components such as synchronous machines, 
exciters, governors, and power system stabilizers. The toolkit is built in Python and leverages the power of 
libraries like torch for efficient computation, making it suitable for both research and educational purposes.

The code is strongly based on [this](https://github.com/hallvar-h/DynPSSimPy) repository, but required a rewrite to 
enable the gradient calculation for optimization purposes. The code is still under development and will be extended 
in the future.

**Note: This repository is still under development and will be extended in the future. Use at your own risk.**

## Features
- **Inherently Parallel Implementation:** A unique and important feature of this simulation framework, as it allows the execution of multiple simulations in parallel by using vectors of parameters for every element.
- **Dynamic Simulation:** Allows for detailed dynamic simulations of power systems, including interactions between various components.
- **Optimization Library:** Features optimizers based on BFGS and automatic differentiation for efficient gradient computation. This part can be used for parameter optimization or identification purposes.
- **Extensible Model Library:** Contains models of AVRs, governors, stabilizers, static models like lines, loads, transformers, and more.
- **Backend Flexibility:** Choose between `torch` and `numpy` as backend for computations.
- **Solver Options:** Includes Euler and Runge Kutta methods for numerical integration.

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:georgkordowich/diffpssi.git
   ```
2. Install the package:
   ```
   pip install diffpssi
   ```
3. Run example simulations:
   ```
   python examples/models/ibb_model/ibb_sim.py
   ```

## Usage and Examples
Detailed usage instructions and examples can be found in the `examples` directory. 

Generally, there are two options to create a simulation. One option is to create the model as a dictionary and pass it 
to the simulation. This is the recommended way. For an example, check out 
the IBB model simulation example under `examples/models/ibb_model/ibb_sim.py` and the corresponding model 
in `examples/models/ibb_model.py`.

The other option is to create the model manually in the simulation file. This can be seen in the example under
`examples/models/ibb_model/ibb_sim_manual.py`. For this option, first a simulation must be created and afterward,
busses, generators, and lines can be added to the simulation. The following code snippet shows how to create
a simulation and add busses, generators, and lines to it.
```python
sim = Pss(parallel_sims=parallel_sims,
          sim_time=10,
          time_step=0.005,
          solver='heun',
          )

sim.fn = 60
sim.base_mva = 2200
sim.base_voltage = 24

sim.add_bus(Bus(name='Bus 0', v_n=24))
sim.add_bus(Bus(name='Bus 1', v_n=24))

sim.add_line(Line(name='L1', from_bus='Bus 0', to_bus='Bus 1', length=1, s_n=2200, v_n=24, unit='p.u.',
                  r=0, x=0.65, b=0, s_n_sys=2200, v_n_sys=24))

sim.add_generator(SynchMachine(name='IBB', bus='Bus 0', s_n=22000, v_n=24, p=-1998, v=0.995, h=3.5e7, d=0,
                               x_d=1.81, x_q=1.76, x_d_t=0.3, x_q_t=0.65, x_d_st=0.23, x_q_st=0.23, t_d0_t=8.0,
                               t_q0_t=1, t_d0_st=0.03, t_q0_st=0.07, f_n_sys=60, s_n_sys=2200, v_n_sys=24))
sim.add_generator(SynchMachine(name='Gen 1', bus='Bus 1', s_n=2200, v_n=24, p=1998, v=1, h=3.5, d=0, x_d=1.81,
                               x_q=1.76, x_d_t=0.3, x_q_t=0.65, x_d_st=0.23, x_q_st=0.23, t_d0_t=8.0, t_q0_t=1,
                               t_d0_st=0.03, t_q0_st=0.07, f_n_sys=60, s_n_sys=2200, v_n_sys=24))

sim.set_slack_bus('Bus 0')
```

Once the model is defined, you can run a simulation. One unique feature of this framework is that you can define
the number of parallel simulations to run. This is useful for parameter optimization, where you can run multiple
simulations in parallel to speed up the process. It is also possible to add events to the simulation, such as
a short circuit event.
```python
sim = mdl.get_model(parallel_sims)
sim.add_sc_event(1, 1.05, 'Bus 1')
sim.set_record_function(record_desired_parameters)

# Run the simulation. Recorder format shall be [batch, timestep, value]
t, recorder = sim.run()
``` 

To acquire and record data during the simulation, you can define a record function. This function is called
at every time step and can be used to record any desired parameters. The function shall return a list of
desired parameters, which will be recorded during the simulation. The recorder format shall be `[batch, timestep, value]`.
By using this function during the simulation, parameters can be plotted afterward.
```python
def record_desired_parameters(simulation):
    # Record the desired parameters
    record_list = [
        simulation.busses[1].models[0].omega.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_d_st.real,
    ]
    return record_list

# Plot the results
plt.figure()
for i in range(len(recorder[0, 0, :])):
    plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
    plt.plot(t, recorder[0, :, i].real)
    plt.ylabel('Parameter {}'.format(i))
    plt.xlabel('Time [s]')
plt.show()
```



## Documentation

For an introductory explanation of how dynamic power system simulations work in principle,
refer to the article: ["Watts Up with Dynamic Power System Simulations"](https://medium.com/@georg.kordowich/watts-up-with-dynamic-power-system-simulations-c0f16fc99769).

## Citing DiffPSSi
If you use DiffPSSi in your research, please cite the following paper: [arXiv:2309.16579](https://arxiv.org/abs/2309.16579).

## Directory Structure

- `models`: Contains example models.
- `optimization_lib`: Includes optimizers and tools for gradient computation.
- `power_sim_lib`: Core library with various submodules:
  - `models`: Models for AVRs, governors, stabilizers, etc.
  - `load_flow`: Tools for load flow analysis.
  - `simulator`: The core simulation class.
  - `solvers`: Numerical solvers for integration.

## Contact
[Contact Information](https://www.ees.tf.fau.de/person/georg-kordowich/)
