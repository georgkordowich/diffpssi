"""
File contains the data for the IBB transformer model.
"""


def load():
    """
    Loads the grid data of the IBB transformer model.
    Returns: The grid data in form of a dictionary.

    """
    return {
        'base_mva': 2200,
        'f': 60,
        'slack_bus': 'Bus 0',
        'base_voltage': 100,

        'busses': [
            ['name', 'V_n'],
            ['Bus 0', 100],
            ['Bus 1', 100],
            ['Bus 2', 10],
        ],

        'lines': [
            ['name', 'from_bus', 'to_bus', 'length', 'unit', 'R', 'X', 'B'],
            ['Line 1', 'Bus 0', 'Bus 1', 1, 'p.u.', 0, 0.22, 0],
        ],

        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['T1', 'Bus 1', 'Bus 2', 2200, 100, 10, 0, 0.15],
        ],

        'generators': {
            'GEN': [
                ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st',
                 'T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st'],
                ['G1', 'Bus 0', 22000, 100, -1998, 0.995, 3.5e7, 0, 1.81, 1.76, 0.3, 0.65, 0.23, 0.23, 8.0, 1, 0.03,
                 0.07],
                ['G2', 'Bus 2', 2200, 10, 1998, 1, 3.5, 0, 1.81, 1.76, 0.3, 0.65, 0.23, 0.23, 8.0, 1, 0.03, 0.07],
            ]
        },
    }
