"""
This example shows how to use the parameter identification tool of PowerFactory to estimate the parameters of a
custom multi-machine model.
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

    sym = prj.GetContents('G2.ElmSym')[0]
    sym_type = sym.typ_id
    sym_type.h = original_params[5] * np.random.uniform(0.5, 2.0)
    sym_type.xd = original_params[6] * np.random.uniform(0.5, 2.0)
    sym_type.xq = original_params[7] * np.random.uniform(0.5, 2.0)
    sym_type.xds = original_params[8] * np.random.uniform(0.5, 2.0)
    sym_type.xqs = original_params[9] * np.random.uniform(0.5, 2.0)

    sym = prj.GetContents('G3.ElmSym')[0]
    sym_type = sym.typ_id
    sym_type.h = original_params[10] * np.random.uniform(0.5, 2.0)
    sym_type.xd = original_params[11] * np.random.uniform(0.5, 2.0)
    sym_type.xq = original_params[12] * np.random.uniform(0.5, 2.0)
    sym_type.xds = original_params[13] * np.random.uniform(0.5, 2.0)
    sym_type.xqs = original_params[14] * np.random.uniform(0.5, 2.0)

    sym = prj.GetContents('G4.ElmSym')[0]
    sym_type = sym.typ_id
    sym_type.h = original_params[15] * np.random.uniform(0.5, 2.0)
    sym_type.xd = original_params[16] * np.random.uniform(0.5, 2.0)
    sym_type.xq = original_params[17] * np.random.uniform(0.5, 2.0)
    sym_type.xds = original_params[18] * np.random.uniform(0.5, 2.0)
    sym_type.xqs = original_params[19] * np.random.uniform(0.5, 2.0)


def get_generator_parameters(prj):
    """
    Function to get the generator parameters from the power factory project.
    Args:
        prj: The power factory project.

    Returns: A list of the generator parameters.

    """
    sym1 = prj.GetContents('G1.ElmSym')[0]
    sym_type1 = sym1.typ_id
    sym2 = prj.GetContents('G2.ElmSym')[0]
    sym_type2 = sym2.typ_id
    sym3 = prj.GetContents('G3.ElmSym')[0]
    sym_type3 = sym3.typ_id
    sym4 = prj.GetContents('G4.ElmSym')[0]
    sym_type4 = sym4.typ_id

    return [
        sym_type1.h,
        sym_type1.xd,
        sym_type1.xq,
        sym_type1.xds,
        sym_type1.xqs,

        sym_type2.h,
        sym_type2.xd,
        sym_type2.xq,
        sym_type2.xds,
        sym_type2.xqs,

        sym_type3.h,
        sym_type3.xd,
        sym_type3.xq,
        sym_type3.xds,
        sym_type3.xqs,

        sym_type4.h,
        sym_type4.xd,
        sym_type4.xq,
        sym_type4.xds,
        sym_type4.xqs,
    ]


def main():
    """
    Main function to run the parameter estimation.
    """
    t_start = time.time()

    pf_path = os.path.abspath(r"data\CustomMultiMachineOpti.pfd")

    prj, sc, param_ident, grid = reset_project(pf_path)

    H_orig_gen1 = 6.0
    X_d_orig_gen1 = 1.81
    X_q_orig_gen1 = 1.76
    X_ds_orig_gen1 = 0.3
    X_qs_orig_gen1 = 0.65

    H_orig_gen2 = 4.9
    X_d_orig_gen2 = 1.81
    X_q_orig_gen2 = 1.76
    X_ds_orig_gen2 = 0.3
    X_qs_orig_gen2 = 0.65

    H_orig_gen3 = 4.1
    X_d_orig_gen3 = 1.81
    X_q_orig_gen3 = 1.76
    X_ds_orig_gen3 = 0.3
    X_qs_orig_gen3 = 0.65

    H_orig_gen4 = 3.2
    X_d_orig_gen4 = 1.81
    X_q_orig_gen4 = 1.76
    X_ds_orig_gen4 = 0.3
    X_qs_orig_gen4 = 0.65

    original_params = [
        H_orig_gen1,
        X_d_orig_gen1,
        X_q_orig_gen1,
        X_ds_orig_gen1,
        X_qs_orig_gen1,

        H_orig_gen2,
        X_d_orig_gen2,
        X_q_orig_gen2,
        X_ds_orig_gen2,
        X_qs_orig_gen2,

        H_orig_gen3,
        X_d_orig_gen3,
        X_q_orig_gen3,
        X_ds_orig_gen3,
        X_qs_orig_gen3,

        H_orig_gen4,
        X_d_orig_gen4,
        X_q_orig_gen4,
        X_ds_orig_gen4,
        X_qs_orig_gen4,
    ]

    for ia in count():
        set_generator_parameters(grid, original_params)

        initial_params = get_generator_parameters(grid)

        start_time = time.time()

        method = 'BFGS'  # 'PSO' or 'BFGS'
        if method == 'PSO':
            param_ident.method = 0
        elif method == 'BFGS':
            param_ident.method = 3

        param_ident.maxNumIter = 10000
        param_ident.maxNumEval = 10000

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
