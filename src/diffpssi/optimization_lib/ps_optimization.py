"""
This file contains the optimization procedure of the power system parameters.
"""
import os
import time
import torch
from matplotlib import pyplot as plt

from diffpssi.optimization_lib.optimizers import CustomBFGSREALOptimizer

# currently only bfgs is supported, as it works best by far
optimizer_dict = {
    'bfgs': CustomBFGSREALOptimizer
}


class PowerSystemOptimization(object):
    """
    This class is used to optimize the parameters of a power system simulation.
    """
    def __init__(self,
                 sim,
                 original_data,
                 params_optimizable,
                 param_names=None,
                 optimizer='bfgs',
                 params_original=None,
                 max_step=0.1,
                 decay=0.9,
                 enable_plots=False,
                 normalize_loss=True,
                 loss_function=None,
                 ):
        """
        :param sim: PowerSystemSimulation object. Used to execute the simulations with the current set of parameters.
        :param original_data: A tensor of original data of the size (batch-size, timesteps, features)
        :param params_optimizable: A list of parameters that should be optimized.
        :param optimizer: The optimizer that should be used (right now only 'bfgs' is supported)
        :param params_original: A list of parameters that can be given for debugging in case they are known
        :param max_step: The relative maximum step size for the optimizer. Can be a list of values (one for each
        parameter), or a single value. Example: 0.1 would mean each parameter can only change by 10% in each step.
        :param decay: The decay factor for the maximum step size. Should be between 0 and 1.
        :param enable_plots: If true, the current best simulation result is plotted after each optimization step.
        :param normalize_loss: If true, the loss function is normalized by the maximum and minimum values of the
        original data. This is useful if the absolute values of the original data are not important, but only the
        relative values.
        :param loss_function: A custom loss function that should be used. If none is given, a default loss function
        is used.

        :return: None
        """
        self.sim = sim

        if sim.backend == 'numpy':
            raise NotImplementedError('Optimization is only supported for the PyTorch backend. '
                                      'Please set the backend to PyTorch in power_sim_lib/backend.py')

        self.target_data = original_data
        self.optimizer = optimizer_dict[optimizer](params_optimizable, max_step=max_step, decay=decay)

        self.params_original = params_original

        self.last_min_loss = None
        if param_names:
            self.param_names = param_names
        else:
            self.param_names = ['Param {}'.format(i) for i in range(len(params_optimizable))]

        self.enable_plots = enable_plots

        self.normalize_loss = normalize_loss

        if loss_function:
            self.loss_function = loss_function
        else:
            # use a default loss function:
            def default_loss_function(sim_result, target_data):
                """
                Calculates the mean absolute error between the simulation result and the target data.
                Args:
                    sim_result: The simulation result of the size (batch-size, timesteps, features)
                    target_data: The target data of the size (batch-size, timesteps, features)

                Returns: A vector of the mean absolute error for each batch element of the size (batch-size)

                """
                # noinspection PyArgumentList
                return torch.mean(torch.sum(torch.abs(target_data - sim_result), dim=2), axis=1)

            self.loss_function = default_loss_function

        if self.normalize_loss:
            # normalize data
            # Determine the minimum and maximum values along dimension 1 of the target data
            self.min_values = torch.min(self.target_data, dim=1)[0].unsqueeze(1)
            self.max_values = torch.max(self.target_data, dim=1)[0].unsqueeze(1)
            self.range_values = self.max_values - self.min_values
            if torch.any(self.range_values == 0):
                # if this does not work, use simulation data to normalize. Note: This is risky and can lead to errors.
                print('Using simulation data to normalize loss function')
            else:
                print('Using target data to normalize loss function')

        # save plot settings for easier comparison
        self.x_lims = None
        self.y_lims = None

    def plot_state(self, t, results, original_data, opt_step):
        """
        Plots the current best simulation result.
        Args:
            t: The timesteps
            results: The simulation results
            original_data: The original data
            opt_step: The current optimization step
        """
        # create as many subplots as we have signals to compare
        plt.figure()

        # set figure size
        plt.gcf().set_size_inches(10, 10)

        for i in range(len(results[0])):
            plt.subplot(len(results[0]), 1, i + 1)
            plt.plot(t, original_data[:, i], label='Original')
            plt.plot(t, results[:, i], label='Simulated', linestyle='--')

            if self.x_lims and self.y_lims:
                plt.xlim(self.x_lims[i])
                plt.ylim(self.y_lims[i])

            plt.ylabel('Signal {}'.format(i))
            plt.xlabel('Time [s]')

        plt.legend()
        plt.savefig('data/plots/optimization_step_{}.png'.format(opt_step))
        # plt.show()

        # get xlims and ylims of all subplots
        if not self.x_lims or not self.y_lims:
            self.x_lims = []
            self.y_lims = []
            for ax in plt.gcf().axes:
                self.x_lims.append(ax.get_xlim())
                self.y_lims.append(ax.get_ylim())

        plt.close()

    def run(self, max_steps=100):
        """
        Runs the previously configured optimization.
        Args:
            max_steps: The maximum number of optimization steps that should be performed.
        """
        if os.environ.get('DIFFPSSI_FORCE_OPT_ITERS') is not None:
            max_steps = int(os.environ.get('DIFFPSSI_FORCE_OPT_ITERS'))
            print('WARNING: FORCING THE USE OF {} OPTIMIZATION ITERATION.'
                  'THIS SHOULD ONLY HAPPEN FOR UNITTESTS'.format(os.environ.get('DIFFPSSI_FORCE_OPT_ITERS')))
        opt_start_time = time.time()

        min_loss_idx = None  # the index of the current best batch element
        results = None  # the simulation results
        t = None  # the timesteps
        opt_step = None  # the current optimization step

        for opt_step in range(max_steps):
            opt_step_start = time.time()
            # set the gradients to zero in order to accumulate the
            self.optimizer.zero_grad()

            # first execute simulation with current parameters
            t, results = self.sim.run()

            if self.normalize_loss:
                if torch.any(self.range_values == 0):
                    min_values = torch.min(results, dim=1)[0].unsqueeze(1)
                    max_values = torch.max(results, dim=1)[0].unsqueeze(1)

                    # take the median of the min and max values along axis 0 to avoid outliers
                    # Also detach the values from the graph, because we will use them for scaling in
                    # later episodes as well
                    self.min_values = torch.median(min_values, dim=0)[0].unsqueeze(0).detach()
                    self.max_values = torch.median(max_values, dim=0)[0].unsqueeze(0).detach()
                    self.range_values = self.max_values - self.min_values

                target_norm = (self.target_data - self.min_values) / self.range_values
                res_norm = (results - self.min_values) / self.range_values
            else:
                target_norm = self.target_data
                res_norm = results

            # then calculate the loss, which corresponds to the mean absolute error
            # For this purpose the sum of all analyzed signals is calculated and the mean along the time axis is taken
            # The result is a vector of the size (batch-size)
            loss = self.loss_function(res_norm, target_norm)

            # take the minimum loss for further analysis
            min_loss_val, min_loss_idx = torch.nan_to_num(loss, 100000).min(dim=0)

            # calculate the gradients for the loss
            loss.sum().backward()

            # perform the optimization step in order to adapt the parameters using the gradients
            self.optimizer.step()

            # print the minimum loss and the corresponding idx
            print('Step: {}, Min. Loss Batch: {}, Min. Loss: {}, Time: {:.2f}s'.format(
                opt_step,
                int(min_loss_idx),
                float(min_loss_val),
                time.time() - opt_step_start)
            )

            # print the current best batch of parameters by comprehending them in a list
            print_list = [p[min_loss_idx].detach() for p in self.optimizer.param_groups[0]['params']]

            print('Current Best Params: ' +
                  ', '.join(['{}: {:.3f}'.format(self.param_names[i], float(print_list[i].data.real)) for i in
                             range(len(print_list))]))

            if self.params_original is not None:
                print('Relative Errors in Percent: ' +
                      ', '.join(['{}: {:.2f}%'.format(self.param_names[i], float(
                          (print_list[i].data.real - self.params_original[i]) / self.params_original[i]) * 100) for i in
                                 range(len(print_list))]))

            print(
                '----------------------------------------------------------------------------------------------------')

            # if all relative errors are smaller than 1% stop the optimization
            if self.params_original is not None:
                if all([abs(float((print_list[i].data.real - self.params_original[i]) / self.params_original[i])) < 0.05
                        for i in range(len(print_list))]):
                    break

            if self.enable_plots:
                plt_original_data = self.target_data[min_loss_idx].detach().numpy()
                plt_results = results[min_loss_idx].detach().numpy()
                self.plot_state(t, plt_results, plt_original_data, opt_step)

            self.sim.reset()

            if self.last_min_loss and self.last_min_loss < min_loss_val:
                self.optimizer.decrease_step_size()

            self.last_min_loss = min_loss_val

        print('Optimization finished in {:.2f} seconds'.format(time.time() - opt_start_time))
        plt_original_data = self.target_data[min_loss_idx].detach().numpy()
        plt_results = results[min_loss_idx].detach().numpy()
        self.plot_state(t, plt_results, plt_original_data, opt_step)
