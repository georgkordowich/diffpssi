"""
Implementation of the Newton-Raphson load flow algorithm for the initialization of the power system simulation.
"""
from itertools import count
import time
from power_sim_lib.models.backend import *


def do_load_flow(ps_sim):
    """
    Performs a load flow calculation for the given power system simulation. The load flow is used to initialize the
    simulation.
    Args:
        ps_sim: The power system simulation to initialize.

    Returns: The calculated complex power at each bus.

    """
    lf_time = time.time()
    pv_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PV']
    pq_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PQ']
    ps_sim.non_slack_busses = [bus for bus in ps_sim.busses if bus.lf_type != 'SL']
    pv_and_pq_bus_ids = pv_bus_ids + pq_bus_ids

    # For both nodes, the desired power is packed into an array
    # The desired power of the IBB is still multiplied by 10 here to get to per unit
    s_soll = torch.stack([bus.get_lf_power() for bus in ps_sim.busses], axis=1)

    # The admittance matrix is necessary for the load flow
    y_bus = ps_sim.admittance_matrix(dynamic=False)

    for ia in count(0):

        v_bus = torch.stack([bus.voltage for bus in ps_sim.busses]).swapaxes(0, 1)

        if y_bus.ndim == 4:
            y_bus = y_bus.squeeze(-1)
        s_calc = v_bus * torch.conj(torch.matmul(y_bus, v_bus))
        y_bus = torch.unsqueeze(y_bus, -1)

        dims_jacobian = len(pv_bus_ids) + 2 * len(pq_bus_ids)

        J = torch.zeros((ps_sim.parallel_sims, dims_jacobian, dims_jacobian), dtype=torch.float64)

        # Construct the Jacobian matrix here
        for i, idx1 in enumerate(pv_and_pq_bus_ids):
            # dP/dTheta
            for k, idx2 in enumerate(pv_and_pq_bus_ids):
                if idx1 == idx2:
                    J[:, i, k] = (abs(v_bus[:, idx1]) ** 2 * y_bus[:, idx1, idx2].imag + s_calc[:, idx1].imag).squeeze()
                else:
                    J[:, i, k] = (
                            -abs(v_bus[:, idx1]) * abs(v_bus[:, idx2]) *
                            (y_bus[:, idx1, idx2].real *
                             torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) -
                             y_bus[:, idx1, idx2].imag *
                             torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()

            # dP/dV
            for k, idx2 in enumerate(pq_bus_ids):
                if idx1 == idx2:
                    J[:, i, k + len(pv_and_pq_bus_ids)] = (-abs(v_bus[:, idx1]) * y_bus[:, idx1, idx2].real -
                                                           s_calc[:, idx1].real / abs(v_bus[:, idx1])).squeeze()
                else:
                    J[:, i, k + len(pv_and_pq_bus_ids)] = (
                            -abs(v_bus[:, idx1]) *
                            (y_bus[:, idx1, idx2].real *
                             torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) +
                             y_bus[:, idx1, idx2].imag *
                             torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()

        for i, idx1 in enumerate(pq_bus_ids):
            # dQ/dTheta
            for k, idx2 in enumerate(pv_and_pq_bus_ids):
                if idx1 == idx2:
                    J[:, i + len(pv_and_pq_bus_ids), k] = (
                            abs(v_bus[:, idx1]) ** 2 * y_bus[:, idx1, idx2].real - s_calc[:, idx1].real).squeeze()
                else:
                    J[:, i + len(pv_and_pq_bus_ids), k] = (
                            abs(v_bus[:, idx1]) * abs(v_bus[:, idx2]) *
                            (y_bus[:, idx1, idx2].real *
                             torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) +
                             y_bus[:, idx1, idx2].imag *
                             torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()

            # dQ/dV
            for k, idx2 in enumerate(pq_bus_ids):
                if idx1 == idx2:
                    J[:, i + len(pv_and_pq_bus_ids), k + len(pv_and_pq_bus_ids)] = (
                            abs(v_bus[:, idx1]) * y_bus[:, idx1, idx2].imag -
                            s_calc[:, idx1].imag / abs(v_bus[:, idx1])).squeeze()
                else:
                    J[:, i + len(pv_and_pq_bus_ids), k + len(pv_and_pq_bus_ids)] = (
                            -abs(v_bus[:, idx1]) *
                            (y_bus[:, idx1, idx2].real *
                             torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) -
                             y_bus[:, idx1, idx2].imag *
                             torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()

        # Calculate the error and the correction vector
        error = s_calc - s_soll
        error = torch.concatenate((error.real[:, pv_bus_ids], error.real[:, pq_bus_ids], error.imag[:, pq_bus_ids]),
                                  axis=1)
        dx = torch.linalg.solve(-J, error)

        # Update the voltages by applying the correction vector to angles and magnitudes
        for i, bus_idx in enumerate(pv_bus_ids):
            ps_sim.busses[bus_idx].update_voltages(abs(ps_sim.busses[bus_idx].voltage) * torch.exp(
                1j * (torch.angle(ps_sim.busses[bus_idx].voltage) - dx[:, i])))
        for i, bus_idx in enumerate(pq_bus_ids):
            ps_sim.busses[bus_idx].update_voltages(abs(ps_sim.busses[bus_idx].voltage) * torch.exp(
                1j * (torch.angle(ps_sim.busses[bus_idx].voltage) - dx[:, i + len(pv_bus_ids)])))
            ps_sim.busses[bus_idx].update_voltages(
                (abs(ps_sim.busses[bus_idx].voltage) - dx[:, i + len(pv_and_pq_bus_ids)]) * torch.exp(
                    1j * torch.angle(ps_sim.busses[bus_idx].voltage)))

        if torch.max(torch.abs(error)) < 1e-8:
            # If the load flow converged, end the loop
            if ps_sim.verbose:
                print('=' * 50)
                print('Load flow finished in {} iterations with error {} in {} seconds'.format(ia, torch.max(
                    torch.abs(error)), time.time() - lf_time))
                print('=' * 50)
            break

        if ia > 600:
            raise ValueError('Load flow did not converge')

    v_bus = torch.stack([bus.voltage for bus in ps_sim.busses], axis=1)
    s_calc = v_bus * torch.conj(torch.matmul(y_bus.squeeze(), v_bus))

    for i, bus in enumerate(ps_sim.busses):
        if bus.lf_type == 'PV':
            bus.update_voltages(v_bus[:, i])

    return s_calc
