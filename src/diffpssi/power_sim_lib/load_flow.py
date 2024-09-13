"""
Implementation of the Newton-Raphson load flow algorithm for the initialization of the power system simulation.
"""
from itertools import count
import time
from src.diffpssi.power_sim_lib.backend import *


def do_load_flow(ps_sim):
    """
    Performs a load flow calculation for the given power system simulation. The load flow is used to initialize the
    simulation. Note: The vectorized version of the load flow is not as easy as the version with "for" loops.
    Scroll down to find the commented out version of the load flow with "for" loops.
    Args:
        ps_sim: The power system simulation to initialize.

    Returns: The calculated complex power at each bus.

    """
    pv_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PV']
    pq_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PQ']
    ps_sim.non_slack_busses = [bus for bus in ps_sim.busses if bus.lf_type != 'SL']
    pv_and_pq_bus_ids = pv_bus_ids + pq_bus_ids

    # For both nodes, the desired power is packed into an array
    # The desired power of the IBB is still multiplied by 10 here to get to per unit

    # Dimensions: [batch, num_busses, 1]
    s_soll = torch.stack([bus.get_lf_power() for bus in ps_sim.busses], axis=1)

    # The admittance matrix is necessary for the load flow
    # Dimensions: [batch, num_busses, num_busses, 1]
    y_bus = ps_sim.lf_admittance_matrix()

    lf_time = time.time()

    for ia in count(0):
        # Dimensions: [batch, num_busses, 1]
        v_bus = torch.stack([bus.voltage for bus in ps_sim.busses]).swapaxes(0, 1)

        if y_bus.ndim == 4:
            y_bus = y_bus.squeeze(-1)

        # dimensions: [batch, num_busses, 1]
        s_calc = v_bus * torch.conj(torch.matmul(y_bus, v_bus))

        # dimensions: [batch, num_busses, num_busses, 1]
        y_bus = torch.unsqueeze(y_bus, -1)

        dims_jacobian = len(pv_bus_ids) + 2 * len(pq_bus_ids)

        J = torch.zeros((ps_sim.parallel_sims, dims_jacobian, dims_jacobian), dtype=torch.float64)

        # Calculate the angles and magnitudes of bus voltages
        v_bus_angle = torch.angle(v_bus)
        v_bus_abs = torch.abs(v_bus)
        # Precompute frequently used variables to avoid redundant slicing
        v_bus_abs_pv_pq = v_bus_abs[:, pv_and_pq_bus_ids]
        v_bus_angle_pv_pq = v_bus_angle[:, pv_and_pq_bus_ids]

        # Reshape tensors for broadcasting
        v_bus_abs_i = torch.unsqueeze(v_bus_abs_pv_pq, 2)
        v_bus_abs_k = torch.unsqueeze(v_bus_abs_pv_pq, 1)
        angle_diff = torch.unsqueeze(v_bus_angle_pv_pq, 2) - torch.unsqueeze(v_bus_angle_pv_pq, 1)

        # Use in-place operations and avoid redundant operations
        y_bus_real_pv_pq = y_bus[:, pv_and_pq_bus_ids][:, :, pv_and_pq_bus_ids].real
        y_bus_imag_pv_pq = y_bus[:, pv_and_pq_bus_ids][:, :, pv_and_pq_bus_ids].imag

        # Preallocate diagonal indices for PV and PQ buses
        diag_indices = torch.arange(len(pv_and_pq_bus_ids))
        diag_indices_pq = torch.arange(len(pq_bus_ids))

        # Block 1: dP/dTheta
        J[:, :len(pv_and_pq_bus_ids), :len(pv_and_pq_bus_ids)] = (
                -v_bus_abs_i * v_bus_abs_k * (
                y_bus_real_pv_pq * torch.sin(angle_diff) - y_bus_imag_pv_pq * torch.cos(angle_diff))
        ).squeeze(-1)

        # Diagonal elements of dP/dTheta
        J[:, diag_indices, diag_indices] = (
                v_bus_abs_pv_pq ** 2 * y_bus[:, pv_and_pq_bus_ids, pv_and_pq_bus_ids].imag +
                s_calc[:, pv_and_pq_bus_ids].imag
        ).squeeze(-1)

        # Block 2: dP/dV (off-diagonal)
        v_bus_abs_pq = v_bus_abs[:, pq_bus_ids]
        # angle_diff_pv_pq = v_bus_angle_pv_pq.unsqueeze(2) - v_bus_angle[:, pq_bus_ids].unsqueeze(1)
        angle_diff_pv_pq = torch.unsqueeze(v_bus_angle_pv_pq, 2) - torch.unsqueeze(v_bus_angle[:, pq_bus_ids], 1)

        y_bus_real_pv_pq_off = y_bus[:, pv_and_pq_bus_ids][:, :, pq_bus_ids].real
        y_bus_imag_pv_pq_off = y_bus[:, pv_and_pq_bus_ids][:, :, pq_bus_ids].imag

        J[:, :len(pv_and_pq_bus_ids), len(pv_and_pq_bus_ids):] = (
                -torch.unsqueeze(v_bus_abs_pv_pq, 2) *
                (y_bus_real_pv_pq_off * torch.cos(angle_diff_pv_pq) +
                 y_bus_imag_pv_pq_off * torch.sin(angle_diff_pv_pq))
        ).squeeze(-1)

        # Diagonal elements of dP/dV
        J[:, diag_indices_pq + len(pv_bus_ids), diag_indices_pq + len(pv_and_pq_bus_ids)] = (
                -v_bus_abs_pq * y_bus[:, pq_bus_ids, pq_bus_ids].real -
                s_calc[:, pq_bus_ids].real / v_bus_abs_pq
        ).squeeze(-1)

        # Block 3: dQ/dTheta (off-diagonal)
        angle_diff_pq_pv_pq = torch.unsqueeze(v_bus_angle[:, pq_bus_ids], 2) - torch.unsqueeze(v_bus_angle_pv_pq, 1)

        y_bus_real_pq_pv_pq = y_bus[:, pq_bus_ids][:, :, pv_and_pq_bus_ids].real
        y_bus_imag_pq_pv_pq = y_bus[:, pq_bus_ids][:, :, pv_and_pq_bus_ids].imag

        J[:, len(pv_and_pq_bus_ids):, :len(pv_and_pq_bus_ids)] = (
                torch.unsqueeze(v_bus_abs_pq, 2) * torch.unsqueeze(v_bus_abs_pv_pq, 1) *
                (y_bus_real_pq_pv_pq * torch.cos(angle_diff_pq_pv_pq) + y_bus_imag_pq_pv_pq * torch.sin(
                    angle_diff_pq_pv_pq))
        ).squeeze(-1)

        # Diagonal elements of dQ/dTheta
        J[:, len(pv_and_pq_bus_ids) + diag_indices_pq, len(pv_bus_ids) + diag_indices_pq] = (
                (v_bus_abs_pq ** 2 * y_bus[:, pq_bus_ids, pq_bus_ids].real) -
                s_calc[:, pq_bus_ids].real
        ).squeeze(-1)

        # Block 4: dQ/dV (off-diagonal)
        angle_diff_pq_pq = torch.unsqueeze(v_bus_angle[:, pq_bus_ids], 2) - torch.unsqueeze(v_bus_angle[:, pq_bus_ids],
                                                                                            1)

        y_bus_real_pq_pq = y_bus[:, pq_bus_ids][:, :, pq_bus_ids].real
        y_bus_imag_pq_pq = y_bus[:, pq_bus_ids][:, :, pq_bus_ids].imag

        J[:, len(pv_and_pq_bus_ids):, len(pv_and_pq_bus_ids):] = (
                -torch.unsqueeze(v_bus_abs_pq, 2) *
                (y_bus_real_pq_pq * torch.sin(angle_diff_pq_pq) - y_bus_imag_pq_pq * torch.cos(angle_diff_pq_pq))
        ).squeeze(-1)

        # Diagonal elements of dQ/dV
        J[:, len(pv_and_pq_bus_ids) + diag_indices_pq, len(pv_and_pq_bus_ids) + diag_indices_pq] = (
                v_bus_abs_pq * y_bus[:, pq_bus_ids, pq_bus_ids].imag -
                s_calc[:, pq_bus_ids].imag / v_bus_abs_pq
        ).squeeze(-1)

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
    s_calc = v_bus * torch.conj(torch.matmul(y_bus.squeeze(-1), v_bus))

    for i, bus in enumerate(ps_sim.busses):
        if bus.lf_type == 'PV':
            bus.update_voltages(v_bus[:, i])

    return s_calc

# def do_load_flow(ps_sim):
#     """
#     Performs a load flow calculation for the given power system simulation. The load flow is used to initialize the
#     simulation.
#     Args:
#         ps_sim: The power system simulation to initialize.
#
#     Returns: The calculated complex power at each bus.
#
#     """
#     lf_time = time.time()
#     pv_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PV']
#     pq_bus_ids = [i for i, bus in enumerate(ps_sim.busses) if bus.lf_type == 'PQ']
#     ps_sim.non_slack_busses = [bus for bus in ps_sim.busses if bus.lf_type != 'SL']
#     pv_and_pq_bus_ids = pv_bus_ids + pq_bus_ids
#
#     # For both nodes, the desired power is packed into an array
#     # The desired power of the IBB is still multiplied by 10 here to get to per unit
#     s_soll = torch.stack([bus.get_lf_power() for bus in ps_sim.busses], axis=1)
#
#     # The admittance matrix is necessary for the load flow
#     y_bus = ps_sim.lf_admittance_matrix()
#
#     for ia in count(0):
#
#         v_bus = torch.stack([bus.voltage for bus in ps_sim.busses]).swapaxes(0, 1)
#
#         if y_bus.ndim == 4:
#             y_bus = y_bus.squeeze(-1)
#         s_calc = v_bus * torch.conj(torch.matmul(y_bus, v_bus))
#         y_bus = torch.unsqueeze(y_bus, -1)
#
#         dims_jacobian = len(pv_bus_ids) + 2 * len(pq_bus_ids)
#
#         J = torch.zeros((ps_sim.parallel_sims, dims_jacobian, dims_jacobian), dtype=torch.float64)
#
#         # Construct the Jacobian matrix here
#         for i, idx1 in enumerate(pv_and_pq_bus_ids):
#             # dP/dTheta
#             for k, idx2 in enumerate(pv_and_pq_bus_ids):
#                 if idx1 == idx2:
#                     J[:, i, k] = (abs(v_bus[:, idx1]) ** 2 * y_bus[:, idx1, idx2].imag + s_calc[:, idx1].imag).squeeze()
#                 else:
#                     J[:, i, k] = (
#                             -abs(v_bus[:, idx1]) * abs(v_bus[:, idx2]) *
#                             (y_bus[:, idx1, idx2].real *
#                              torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) -
#                              y_bus[:, idx1, idx2].imag *
#                              torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()
#
#             # dP/dV
#             for k, idx2 in enumerate(pq_bus_ids):
#                 if idx1 == idx2:
#                     J[:, i, k + len(pv_and_pq_bus_ids)] = (-abs(v_bus[:, idx1]) * y_bus[:, idx1, idx2].real -
#                                                            s_calc[:, idx1].real / abs(v_bus[:, idx1])).squeeze()
#                 else:
#                     J[:, i, k + len(pv_and_pq_bus_ids)] = (
#                             -abs(v_bus[:, idx1]) *
#                             (y_bus[:, idx1, idx2].real *
#                              torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) +
#                              y_bus[:, idx1, idx2].imag *
#                              torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()
#
#         for i, idx1 in enumerate(pq_bus_ids):
#             # dQ/dTheta
#             for k, idx2 in enumerate(pv_and_pq_bus_ids):
#                 if idx1 == idx2:
#                     J[:, i + len(pv_and_pq_bus_ids), k] = (
#                             abs(v_bus[:, idx1]) ** 2 * y_bus[:, idx1, idx2].real - s_calc[:, idx1].real).squeeze()
#                 else:
#                     J[:, i + len(pv_and_pq_bus_ids), k] = (
#                             abs(v_bus[:, idx1]) * abs(v_bus[:, idx2]) *
#                             (y_bus[:, idx1, idx2].real *
#                              torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) +
#                              y_bus[:, idx1, idx2].imag *
#                              torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()
#
#             # dQ/dV
#             for k, idx2 in enumerate(pq_bus_ids):
#                 if idx1 == idx2:
#                     J[:, i + len(pv_and_pq_bus_ids), k + len(pv_and_pq_bus_ids)] = (
#                             abs(v_bus[:, idx1]) * y_bus[:, idx1, idx2].imag -
#                             s_calc[:, idx1].imag / abs(v_bus[:, idx1])).squeeze()
#                 else:
#                     J[:, i + len(pv_and_pq_bus_ids), k + len(pv_and_pq_bus_ids)] = (
#                             -abs(v_bus[:, idx1]) *
#                             (y_bus[:, idx1, idx2].real *
#                              torch.sin(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])) -
#                              y_bus[:, idx1, idx2].imag *
#                              torch.cos(torch.angle(v_bus[:, idx1]) - torch.angle(v_bus[:, idx2])))).squeeze()
#
#         # Calculate the error and the correction vector
#         error = s_calc - s_soll
#         error = torch.concatenate((error.real[:, pv_bus_ids], error.real[:, pq_bus_ids], error.imag[:, pq_bus_ids]),
#                                   axis=1)
#         dx = torch.linalg.solve(-J, error)
#
#         if ia == 1:
#             print(J[0])
#             raise ValueError('Stop')
#
#         # Update the voltages by applying the correction vector to angles and magnitudes
#         for i, bus_idx in enumerate(pv_bus_ids):
#             ps_sim.busses[bus_idx].update_voltages(abs(ps_sim.busses[bus_idx].voltage) * torch.exp(
#                 1j * (torch.angle(ps_sim.busses[bus_idx].voltage) - dx[:, i])))
#         for i, bus_idx in enumerate(pq_bus_ids):
#             ps_sim.busses[bus_idx].update_voltages(abs(ps_sim.busses[bus_idx].voltage) * torch.exp(
#                 1j * (torch.angle(ps_sim.busses[bus_idx].voltage) - dx[:, i + len(pv_bus_ids)])))
#             ps_sim.busses[bus_idx].update_voltages(
#                 (abs(ps_sim.busses[bus_idx].voltage) - dx[:, i + len(pv_and_pq_bus_ids)]) * torch.exp(
#                     1j * torch.angle(ps_sim.busses[bus_idx].voltage)))
#
#         if torch.max(torch.abs(error)) < 1e-8:
#             # If the load flow converged, end the loop
#             if ps_sim.verbose:
#                 print('=' * 50)
#                 print('Load flow finished in {} iterations with error {} in {} seconds'.format(ia, torch.max(
#                     torch.abs(error)), time.time() - lf_time))
#                 print('=' * 50)
#             break
#
#         if ia > 600:
#             raise ValueError('Load flow did not converge')
#
#     v_bus = torch.stack([bus.voltage for bus in ps_sim.busses], axis=1)
#     s_calc = v_bus * torch.conj(torch.matmul(y_bus.squeeze(), v_bus))
#
#     for i, bus in enumerate(ps_sim.busses):
#         if bus.lf_type == 'PV':
#             bus.update_voltages(v_bus[:, i])
#
#     return s_calc
