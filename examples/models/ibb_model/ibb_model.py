"""
This file contains the data of a simple infinite bus bar model.
"""


def load():
    """
    This function returns a dictionary that contains the data for the simple infinite bus bar model.
    Returns: The data in the form of a dictionary.

    """
    return {
        'base_mva': 2200,
        'f': 60,
        'slack_bus': 'Bus 0',
        'base_voltage': 24,

        'busses': [
            ['name', 'V_n'],
            ['Bus 0', 24],
            ['Bus 1', 24],
        ],

        'lines': [
            ['name', 'from_bus', 'to_bus', 'length', 'S_n', 'V_n', 'unit', 'R', 'X', 'B'],
            ['L1', 'Bus 0', 'Bus 1', 1, 2200, 24, 'p.u.', 0, 0.65, 0],
        ],

        'generators': {
            'GEN': [
                ['name', 'bus', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'X_d', 'X_q', 'X_d_t', 'X_q_t', 'X_d_st', 'X_q_st',
                 'T_d0_t', 'T_q0_t', 'T_d0_st', 'T_q0_st'],
                ['IBB', 'Bus 0', 22000, 24, -1998, 0.995, 3.5e7, 0, 1.81, 1.76, 0.3, 0.65, 0.23, 0.23, 8.0, 1, 0.03,
                 0.07],
                ['Gen 1', 'Bus 1', 2200, 24, 1998, 1, 3.5, 0, 1.81, 1.76, 0.3, 0.65, 0.23, 0.23, 8.0, 1, 0.03, 0.07],
            ],
        },
    }
