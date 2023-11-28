from power_sim_lib.simulator import PowerSystemSimulation as Pss


gen1_dict = {
    'name': 'Gen 1',
    's_n': 247.5,
    'v_n': 16.5,
    'p_soll_mw': 71.6,
    'v_soll': 1.04,
    'h': 9.551516,
    'd': 0,
    'x_d': 0.36135,
    'x_q': 0.2398275,
    'x_d_t': 0.15048,
    'x_q_t': 0.15048,
    'x_d_st': 0.1,
    'x_q_st': 0.1,
    't_d0_t': 8.96,
    't_q0_t': 1,
    't_d0_st': 0.075,
    't_q0_st': 0.15,
}
gen2_dict = {
    'name': 'Gen 2',
    's_n': 192,
    'v_n': 18,
    'p_soll_mw': 163.,
    'v_soll': 1.025,
    'h': 3.921568,
    'd': 0,
    'x_d': 1.719936,
    'x_q': 1.65984,
    'x_d_t': 0.230016,
    'x_q_t': 0.378048,
    'x_d_st': 0.2,
    'x_q_st': 0.2,
    't_d0_t': 6.0,
    't_q0_t': 0.535,
    't_d0_st': 0.0575,
    't_q0_st': 0.0945,
}
gen3_dict = {
    'name': 'Gen 3',
    's_n': 128,
    'v_n': 13.8,
    'p_soll_mw': 85.,
    'v_soll': 1.025,
    'h': 2.766544,
    'd': 0,
    'x_d': 1.68,
    'x_q': 1.609984,
    'x_d_t': 0.232064,
    'x_q_t': 0.32,
    'x_d_st': 0.2,
    'x_q_st': 0.2,
    't_d0_t': 5.89,
    't_q0_t': 0.6,
    't_d0_st': 0.0575,
    't_q0_st': 0.08,
}


def get_model(parallel_sims):
    # Create a new simulator object
    sim = Pss(parallel_sims=parallel_sims, base_mva=250, fn=60, base_voltage=230, sim_time=10)
    sim.add_bus(name='Bus 1', lf_type='SL', v_n=16.5)
    sim.add_bus(name='Bus 2', lf_type='PV', v_n=18)
    sim.add_bus(name='Bus 3', lf_type='PV', v_n=13.8)
    sim.add_bus(name='Bus 4', lf_type='PQ', v_n=230)
    sim.add_bus(name='Bus 5', lf_type='PQ', v_n=230)
    sim.add_bus(name='Bus 6', lf_type='PQ', v_n=230)
    sim.add_bus(name='Bus 7', lf_type='PQ', v_n=230)
    sim.add_bus(name='Bus 8', lf_type='PQ', v_n=230)
    sim.add_bus(name='Bus 9', lf_type='PQ', v_n=230)

    sim.add_trafo('Bus 4', 'Bus 1', s_n=250, r=0, x=0.144, v_n_from=230, v_n_to=13.8)
    sim.add_trafo('Bus 7', 'Bus 2', s_n=200, r=0, x=0.125, v_n_from=230, v_n_to=18)
    sim.add_trafo('Bus 9', 'Bus 3', s_n=150, r=0, x=0.0879, v_n_from=230, v_n_to=16.5)

    sim.add_line('Bus 4', 'Bus 5', r=5.29, x=44.965, b=332.7e-6, length=1, unit='Ohm')
    sim.add_line('Bus 4', 'Bus 6', r=8.993, x=48.668, b=298.69e-6, length=1, unit='Ohm')
    sim.add_line('Bus 5', 'Bus 7', r=16.928, x=85.169, b=578.45e-6, length=1, unit='Ohm')
    sim.add_line('Bus 6', 'Bus 9', r=20.631, x=89.93, b=676.75e-6, length=1, unit='Ohm')
    sim.add_line('Bus 7', 'Bus 8', r=4.4965, x=38.088, b=281.66e-6, length=1, unit='Ohm')
    sim.add_line('Bus 8', 'Bus 9', r=6.2951, x=53.3232, b=395.08e-6, length=1, unit='Ohm')

    sim.add_load('Bus 5', p_soll_mw=125, q_soll_mvar=50)
    sim.add_load('Bus 6', p_soll_mw=90, q_soll_mvar=30)
    sim.add_load('Bus 8', p_soll_mw=100, q_soll_mvar=35)

    sim.add_generator('Bus 1', gen1_dict)
    sim.add_generator('Bus 2', gen2_dict)
    sim.add_generator('Bus 3', gen3_dict)

    return sim
