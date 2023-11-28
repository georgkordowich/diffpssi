import time

import torch
from matplotlib import pyplot as plt

from optimization_lib.optimizers import CustomBFGSREALOptimizer

optimizer_dict = {
    'bfgs': CustomBFGSREALOptimizer
}

class PowerSystemOptimization(object):
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
    ):
        """
        :param sim: PowerSystemSimulation object
        :param original_data: A tensor of original data of the size (batchsize, timesteps, features)
        :param params_optimizable: A list of parameters that should be optimized
        :param optimizer: The optimizer that should be used (right now only 'bfgs' is supported)
        :param params_original: A list of parameters that can be given for debugging in case they are known
        """
        self.sim = sim
        self.original_data = original_data
        self.optimizer = optimizer_dict[optimizer](params_optimizable, max_step=max_step, decay=decay)

        self.params_original = params_original

        self.last_min_loss = None
        if param_names:
            self.param_names = param_names
        else:
            self.param_names = ['Param {}'.format(i) for i in range(len(params_optimizable))]

        self.enable_plots = enable_plots

        self.normalize_loss = normalize_loss

    def plot_state(self, t, results, original_data):
        # create as many subplots as we have signals to compare
        plt.figure()

        for i in range(len(results[0])):
            plt.subplot(len(results[0]), 1, i + 1)
            plt.plot(t, original_data[:, i], label='Original')
            plt.plot(t, results[:, i], label='Simulated', linestyle='--')

            plt.ylabel('Signal {}'.format(i))
            plt.xlabel('Time [s]')

        plt.legend()
        plt.show()

    def run(self, max_steps=100):
        opt_start_time = time.time()
        for opt_step in range(max_steps):
            opt_step_start = time.time()
            # set the gradients to zero in order to accumulate the
            self.optimizer.zero_grad()

            # first execute simulation with current parameters
            t, results = self.sim.run()

            if self.normalize_loss:
                # normalize data
                # Determine the minimum and maximum values along dimension 1
                min_values = torch.min(self.original_data, dim=1)[0].unsqueeze(1)
                max_values = torch.max(self.original_data, dim=1)[0].unsqueeze(1)

                # Calculate the range for scaling
                range_values = max_values - min_values

                orig_norm = (self.original_data - min_values) / range_values
                res_norm = (results - min_values) / range_values

            else:
                orig_norm = self.original_data
                res_norm = results

            # then calculate the loss, which corresponds to the mean absolute error
            # For this purpose the sum of all analyzed signals is calculated and the the mean along the time axis is taken
            # The result is a vector of the size (batchsize)
            loss = torch.mean(torch.sum(torch.abs(orig_norm- res_norm), dim=2), axis=1)

            # take the minimum loss for further analysis
            min_loss_val, min_loss_idx = torch.nan_to_num(loss, 100000).min(dim=0)

            # calculate the gradients for the loss
            loss.sum().backward()

            # perform the optimization step in order to adapt the parameters using the gradients
            self.optimizer.step()

            # print(int(min_loss_idx), float(min_loss_val))
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
                  ', '.join(['{}: {:.3f}'.format(self.param_names[i], float(print_list[i].data.real)) for i in range(len(print_list))]))

            if self.params_original is not None:
                print('Relative Errors in Percent: ' +
                      ', '.join(['{}: {:.2f}%'.format(self.param_names[i], float((print_list[i].data.real - self.params_original[i]) / self.params_original[i])*100) for i in range(len(print_list))]))

            print('----------------------------------------------------------------------------------------------------')

            # if all relative errors are smaller than 1% stop the optimization
            if self.params_original is not None:
                if all([abs(float((print_list[i].data.real - self.params_original[i]) / self.params_original[i])) < 0.01 for i in range(len(print_list))]):
                    print('Optimization finished in {:.2f} seconds'.format(time.time() - opt_start_time))
                    plt_original_data = self.original_data[min_loss_idx].detach().numpy()
                    plt_results = results[min_loss_idx].detach().numpy()
                    self.plot_state(t, plt_results, plt_original_data)
                    break

            if self.enable_plots:
                plt_original_data = self.original_data[min_loss_idx].detach().numpy()
                plt_results = results[min_loss_idx].detach().numpy()
                self.plot_state(t, plt_results, plt_original_data)

            self.sim.reset()

            if self.last_min_loss and self.last_min_loss < min_loss_val:
                self.optimizer.decrease_step_size()

            self.last_min_loss = min_loss_val
