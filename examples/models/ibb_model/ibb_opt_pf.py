"""
This example shows how to use the power factory API to optimize the parameters of a power factory model.
"""
import os
from itertools import count

import numpy as np
from pf_util.pf_tools import reset_project
import time

np.random.seed(0)


def set_generator_parameters(prj, original_params):
    """
    Function to set the generator parameters in the power factory project randomly, so they can be optimized.

    Args:
        prj: The power factory project.
        original_params: The original parameters of the generators.
    """
    sym = prj.GetContents('G1.ElmSym')[0]
    sym_type = sym.typ_id

    sym_type.h = original_params[0] * np.random.uniform(0.5, 2.0)
    sym_type.xd = original_params[1] * np.random.uniform(0.5, 2.0)
    sym_type.xq = original_params[2] * np.random.uniform(0.5, 2.0)
    sym_type.xds = original_params[3] * np.random.uniform(0.5, 2.0)
    sym_type.xqs = original_params[4] * np.random.uniform(0.5, 2.0)


def get_generator_parameters(prj):
    """
    Function to get the generator parameters from the power factory project.
    Args:
        prj: The power factory project.

    Returns: A list of the generator parameters.

    """
    sym = prj.GetContents('G1.ElmSym')[0]
    sym_type = sym.typ_id

    return [
        sym_type.h,
        sym_type.xd,
        sym_type.xq,
        sym_type.xds,
        sym_type.xqs,
    ]


def main():
    """
    Main function to run the parameter estimation.
    """
    t_start = time.time()

    pf_path = os.path.abspath(r"data\SMIBKundurOptim.pfd")

    prj, sc, param_ident, grid = reset_project(pf_path)
    method = 'PSO'  # 'PSO' or 'BFGS'

    H_gen_orig = 3.5
    X_d_orig = 1.81
    X_q_orig = 1.76
    X_ds_orig = 0.3
    X_qs_orig = 0.65

    original_params = [
        H_gen_orig,
        X_d_orig,
        X_q_orig,
        X_ds_orig,
        X_qs_orig,
    ]

    for ia in count():
        set_generator_parameters(grid, original_params)

        initial_params = get_generator_parameters(grid)

        start_time = time.time()

        if method == 'PSO':
            param_ident.method = 0
        elif method == 'BFGS':
            param_ident.method = 3

        param_ident.maxNumIter = 1000
        param_ident.maxNumEval = 1000

        param_ident.Execute()
        end_time = time.time()

        param_array = get_generator_parameters(grid)

        rel_errors = (np.array(param_array) - np.array(original_params)) * 100 / np.array(original_params)

        print('Parameter Estimation step {} finished in {:.2f} seconds'.format(ia, end_time - start_time))
        print('Initial Params: ', ['%.3f' % elem for elem in initial_params])
        print('Abs. Data: ', ['%.3f' % elem for elem in param_array])
        print('Rel. Errors: ', ['%.3f' % elem for elem in rel_errors])
        print('----------------------------------------------------------------------------------------------------')

        if all(abs(i) < 1 for i in rel_errors):
            print('Optimization finished in {:.2f} seconds'.format(time.time() - t_start))
            break


if __name__ == '__main__':
    main()
