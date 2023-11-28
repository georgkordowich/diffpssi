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

sexs_dict = {
    'name': 'SEXS Gen 1',
    'k': 100,
    't_a': 4.0,
    't_b': 10.0,
    't_e': 0.1,
    'e_min': -3.0,
    'e_max': 3.0,
}

gov_dict = {
    'name': 'Gov Gen 1',
    'r': 0.05,
    'd_t': 0.02,
    'v_min': 0.0,
    'v_max': 1.0,
    't_1': 0.5,
    't_2': 2.0,
    't_3': 2.0,
}

pss_dict = {
    'name': 'PSS Gen 1',
    'k_w': 50,
    't_w': 10,
    't_1': 0.5,
    't_2': 0.05,
    't_3': 0.5,
    't_4': 0.05,
    'h_lim': 0.03,
}


def get_model(parallel_sims):
    # Create a new simulator object
    sim = Pss(parallel_sims=parallel_sims, base_mva=2200, fn=60, base_voltage=24, sim_time=10)
    sim.add_bus(name='Bus 0', lf_type='SL', v_n=24)
    sim.add_bus(name='Bus 1', lf_type='PV', v_n=24)

    # Add a model to the second bus
    sim.add_generator('Bus 0', ibb_dict)
    sim.add_generator('Bus 1', gen1_dict)

    sim.busses[sim.bus_names['Bus 1']].models[0].add_exciter(sexs_dict)
    sim.busses[sim.bus_names['Bus 1']].models[0].add_governor(gov_dict)
    sim.busses[sim.bus_names['Bus 1']].models[0].add_pss(pss_dict)

    sim.add_line('Bus 0', 'Bus 1', r=0, x=0.65, b=0, length=1, unit='pu')

    return sim
