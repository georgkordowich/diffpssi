"""
File contains an example of how to simulate the K2A model.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.diffpssi.power_sim_lib.simulator import PowerSystemSimulation as Pss

from src.diffpssi.power_sim_lib.models.synchronous_machine import SynchMachine
from src.diffpssi.power_sim_lib.models.static_models import *
from src.diffpssi.power_sim_lib.models.exciters import SEXS
from src.diffpssi.power_sim_lib.models.governors import TGOV1
from src.diffpssi.power_sim_lib.models.stabilizers import STAB1


def record_desired_parameters(simulation):
    """
    Records the desired parameters of the simulation.
    Args:
        simulation: The simulation to record the parameters from.

    Returns: A list of the recorded parameters.

    """
    # Record the desired parameters
    record_list = [
        simulation.busses[0].models[0].delta.real,
        simulation.busses[1].models[0].delta.real,
        simulation.busses[2].models[0].delta.real,
        simulation.busses[3].models[0].delta.real,

        simulation.busses[0].models[0].omega.real,
        simulation.busses[1].models[0].omega.real,
        simulation.busses[2].models[0].omega.real,
        simulation.busses[3].models[0].omega.real,

        simulation.busses[0].models[0].e_q_st.real,
        simulation.busses[1].models[0].e_q_st.real,
        simulation.busses[2].models[0].e_q_st.real,
        simulation.busses[3].models[0].e_q_st.real,

        simulation.busses[0].models[0].e_d_st.real,
        simulation.busses[1].models[0].e_d_st.real,
        simulation.busses[2].models[0].e_d_st.real,
        simulation.busses[3].models[0].e_d_st.real,
    ]
    return record_list


def main():
    """
    This function simulates the K2A model.
    """
    parallel_sims = 1

    sim = Pss(parallel_sims=parallel_sims,
              sim_time=10,
              time_step=0.005,
              solver='heun',
              )

    sim.fn = 50
    sim.base_mva = 900
    sim.base_voltage = 230

    sim.add_bus(Bus(name='B1', v_n=20))
    sim.add_bus(Bus(name='B2', v_n=20))
    sim.add_bus(Bus(name='B3', v_n=20))
    sim.add_bus(Bus(name='B4', v_n=20))
    sim.add_bus(Bus(name='B5', v_n=230))
    sim.add_bus(Bus(name='B6', v_n=230))
    sim.add_bus(Bus(name='B7', v_n=230))
    sim.add_bus(Bus(name='B8', v_n=230))
    sim.add_bus(Bus(name='B9', v_n=230))
    sim.add_bus(Bus(name='B10', v_n=230))
    sim.add_bus(Bus(name='B11', v_n=230))

    sim.add_line(Line(name='L5-6', from_bus='B5', to_bus='B6', length=25, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L6-7', from_bus='B6', to_bus='B7', length=10, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L7-8-1', from_bus='B7', to_bus='B8', length=110, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L7-8-2', from_bus='B7', to_bus='B8', length=110, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L8-9-1', from_bus='B8', to_bus='B9', length=110, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L8-9-2', from_bus='B8', to_bus='B9', length=110, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L9-10', from_bus='B9', to_bus='B10', length=10, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))
    sim.add_line(Line(name='L10-11', from_bus='B10', to_bus='B11', length=25, s_n=100, v_n=230, unit='p.u.',
                      r=1e-4, x=1e-3, b=1.75e-3, s_n_sys=900, v_n_sys=230))

    sim.add_transformer(Transformer(name='T1', from_bus='B1', to_bus='B5', s_n=900, v_n_from=20, v_n_to=230, r=0,
                                    x=0.15, s_n_sys=900))
    sim.add_transformer(Transformer(name='T2', from_bus='B2', to_bus='B6', s_n=900, v_n_from=20, v_n_to=230, r=0,
                                    x=0.15, s_n_sys=900))
    sim.add_transformer(Transformer(name='T3', from_bus='B3', to_bus='B11', s_n=900, v_n_from=20, v_n_to=230, r=0,
                                    x=0.15, s_n_sys=900))
    sim.add_transformer(Transformer(name='T4', from_bus='B4', to_bus='B10', s_n=900, v_n_from=20, v_n_to=230,
                                    r=0, x=0.15, s_n_sys=900))

    sim.add_load(Load(name='L1', bus='B7', p=967, q=100, model='Z', s_n_sys=900))
    sim.add_load(Load(name='L2', bus='B9', p=1767, q=100, model='Z', s_n_sys=900))

    sim.add_shunt(Shunt(name='C1', bus='B7', v_n=230, q=200, model='Z', s_n_sys=900))
    sim.add_shunt(Shunt(name='C2', bus='B9', v_n=230, q=350, model='Z', s_n_sys=900))

    sim.add_generator(SynchMachine(name='G1', bus='B1', s_n=900, v_n=20, p=700, v=1.03, h=6.5, d=0, x_d=1.8,
                                   x_q=1.7, x_d_t=0.3, x_q_t=0.55, x_d_st=0.25, x_q_st=0.25, t_d0_t=8.0, t_q0_t=0.4,
                                   t_d0_st=0.03, t_q0_st=0.05, f_n_sys=50, s_n_sys=900, v_n_sys=230))
    sim.add_generator(SynchMachine(name='G2', bus='B2', s_n=900, v_n=20, p=700, v=1.01, h=6.5, d=0, x_d=1.8,
                                   x_q=1.7, x_d_t=0.3, x_q_t=0.55, x_d_st=0.25, x_q_st=0.25, t_d0_t=8.0, t_q0_t=0.4,
                                   t_d0_st=0.03, t_q0_st=0.05, f_n_sys=50, s_n_sys=900, v_n_sys=230))
    sim.add_generator(SynchMachine(name='G3', bus='B3', s_n=900, v_n=20, p=719, v=1.03, h=6.175, d=0, x_d=1.8,
                                   x_q=1.7, x_d_t=0.3, x_q_t=0.55, x_d_st=0.25, x_q_st=0.25, t_d0_t=8.0, t_q0_t=0.4,
                                   t_d0_st=0.03, t_q0_st=0.05, f_n_sys=50, s_n_sys=900, v_n_sys=230))
    sim.add_generator(SynchMachine(name='G4', bus='B4', s_n=900, v_n=20, p=700, v=1.01, h=6.175, d=0, x_d=1.8,
                                   x_q=1.7, x_d_t=0.3, x_q_t=0.55, x_d_st=0.25, x_q_st=0.25, t_d0_t=8.0, t_q0_t=0.4,
                                   t_d0_st=0.03, t_q0_st=0.05, f_n_sys=50, s_n_sys=900, v_n_sys=230))

    sim.add_governor(TGOV1(name='GOV1', gen='G1', r=0.05, d_t=0.02, v_min=0, v_max=1, t_1=0.5, t_2=1, t_3=2))
    sim.add_governor(TGOV1(name='GOV2', gen='G2', r=0.05, d_t=0.02, v_min=0, v_max=1, t_1=0.5, t_2=1, t_3=2))
    sim.add_governor(TGOV1(name='GOV3', gen='G3', r=0.05, d_t=0.02, v_min=0, v_max=1, t_1=0.5, t_2=1, t_3=2))
    sim.add_governor(TGOV1(name='GOV4', gen='G4', r=0.05, d_t=0.02, v_min=0, v_max=1, t_1=0.5, t_2=1, t_3=2))

    sim.add_exciter(SEXS(name='AVR1', gen='G1', k=100, t_a=2.0, t_b=10.0, t_e=0.1, e_min=-3, e_max=3))
    sim.add_exciter(SEXS(name='AVR2', gen='G2', k=100, t_a=2.0, t_b=10.0, t_e=0.1, e_min=-3, e_max=3))
    sim.add_exciter(SEXS(name='AVR3', gen='G3', k=100, t_a=2.0, t_b=10.0, t_e=0.1, e_min=-3, e_max=3))
    sim.add_exciter(SEXS(name='AVR4', gen='G4', k=100, t_a=2.0, t_b=10.0, t_e=0.1, e_min=-3, e_max=3))

    sim.add_pss(STAB1(name='PSS1', gen='G1', k=50, t=10.0, t_1=0.5, t_2=0.5, t_3=0.05, t_4=0.05, h_lim=0.03))
    sim.add_pss(STAB1(name='PSS2', gen='G2', k=50, t=10.0, t_1=0.5, t_2=0.5, t_3=0.05, t_4=0.05, h_lim=0.03))
    sim.add_pss(STAB1(name='PSS3', gen='G3', k=50, t=10.0, t_1=0.5, t_2=0.5, t_3=0.05, t_4=0.05, h_lim=0.03))
    sim.add_pss(STAB1(name='PSS4', gen='G4', k=50, t=10.0, t_1=0.5, t_2=0.5, t_3=0.05, t_4=0.05, h_lim=0.03))

    sim.set_slack_bus('B3')

    sim.add_sc_event(1, 1.1, 'B1')

    sim.set_record_function(record_desired_parameters)
    t, recorder = sim.run()

    # Format shall be [batch, timestep, value]
    # create a new subplot for each parameter
    plt.figure()

    for i in range(len(recorder[0, 0, :])):
        plt.subplot(len(recorder[0, 0, :]), 1, i + 1)
        plt.plot(t, recorder[0, :, i].real)
        plt.ylabel('Parameter {}'.format(i))
        plt.xlabel('Time [s]')

    plt.savefig('data/plots/original.png'.format())
    plt.show()

    # add time to the first column
    saver = np.zeros((len(t), len(recorder[0, 0, :]) + 1))
    saver[:, 0] = t
    saver[:, 1:] = recorder[0, :, :].real

    np.save('data/original_data_t.npy', saver)
    np.save('data/original_data.npy', recorder[0].real)


if __name__ == '__main__':
    main()
