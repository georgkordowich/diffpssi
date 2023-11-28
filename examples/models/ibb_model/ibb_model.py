from power_sim_lib.simulator import PowerSystemSimulation as Pss

ibb_dict = {
    'name': 'IBB 1',
    's_n': 22000,
    'v_n': 24,
    'p_soll_mw': -1998,
    'v_soll': 0.995,
    'h': 3.5e7,
    'd': 0,
    'x_d': 1.81,
    'x_q': 1.76,
    'x_d_t': 0.3,
    'x_q_t': 0.65,
    'x_d_st': 0.23,
    'x_q_st': 0.23,
    't_d0_t': 8.0,
    't_q0_t': 1,
    't_d0_st': 0.03,
    't_q0_st': 0.07
}
gen1_dict = {
    'name': 'Gen 1',
    's_n': 2200,
    'v_n': 24,
    'p_soll_mw': 1998,
    'v_soll': 1,
    'h': 3.5,
    'd': 0,
    'x_d': 1.81,
    'x_q': 1.76,
    'x_d_t': 0.3,
    'x_q_t': 0.65,
    'x_d_st': 0.23,
    'x_q_st': 0.23,
    't_d0_t': 8.0,
    't_q0_t': 1,
    't_d0_st': 0.03,
    't_q0_st': 0.07
}


def get_model(parallel_sims):
    # Create a new simulator object
    sim = Pss(fn=60,
              base_mva=2200,
              base_voltage=24,
              sim_time=10,
              parallel_sims=parallel_sims)
    sim.add_bus(name='Bus 0', lf_type='SL', v_n=24)
    sim.add_bus(name='Bus 1', lf_type='PV', v_n=24)

    # Add a model to the second bus
    sim.add_generator('Bus 0', ibb_dict)
    sim.add_generator('Bus 1', gen1_dict)

    sim.add_line('Bus 0', 'Bus 1', r=0, x=0.65, b=0, length=1, unit='pu')

    return sim
