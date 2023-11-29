# PSIPSY: A Python based framework for dynamic simulation and optimization of power systems
## Overview
PSIPSY contains a framework designed for simulating and optimizing the dynamic behavior 
of power systems. It includes detailed models of various power system components such as synchronous machines, 
exciters, governors, and power system stabilizers. The toolkit is built in Python and leverages the power of 
libraries like torch for efficient computation, making it suitable for both research and educational purposes.

The code is strongly based on [this](https://github.com/hallvar-h/DynPSSimPy) repository, but required a rewrite to 
enable the gradient calculation for optimization purposes. The code is still under development and will be extended in the future.

## Features
- **Dynamic Simulation:** Allows for detailed dynamic simulations of power systems, including interactions between various components.
- **Optimization Library:** Features optimizers based on BFGS and automatic differentiation for efficient gradient computation. This part can be used for parameter optimization or identification purposes.
- **Extensible Model Library:** Contains models of AVRs, governors, stabilizers, static models like lines, loads, transformers, and more.
- **Backend Flexibility:** Choose between `torch` and `numpy` as backend for computations.
- **Solver Options:** Includes Euler and Heun methods for numerical integration.

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:georgkordowich/psipsy.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run example simulations:
   ```
   python examples/models/ibb_model/ibb_sim.py
   ```

## Usage and Examples

Detailed usage instructions and examples can be found in the `examples` directory. For a guided introduction, check out the IBB model simulation example under `examples/models/ibb_model/ibb_sim.py` and then corresponding model in `examples/models/ibb_model.py`.

## Documentation

For an in-depth explanation of how dynamic power system simulations work in principle, refer to the article: ["Watts Up with Dynamic Power System Simulations"](https://medium.com/@georg.kordowich/watts-up-with-dynamic-power-system-simulations-c0f16fc99769).

## Citing PSIPSY
If you use PSIPSY in your research, please cite the following paper: [arXiv:2309.16579](https://arxiv.org/abs/2309.16579).

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
