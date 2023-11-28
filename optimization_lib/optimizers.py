import torch

class CustomBFGSREALOptimizer(torch.optim.Optimizer):
    """
    Optimizer that reduces the learning rate each step, automatically choses a reasonable learning rate and enforces
    a maximum relative step size for the parameter optimization.
    """
    # Init Method:
    def __init__(self, params, max_step=0.1, decay=0.9, verbose=True):
        self.decay = decay
        # Max step can be a list of values (one for each parameter), or a single value.
        try:
            max_steps = [max_step[i] * abs(p.data.real) for i, p in enumerate(params)]
        except TypeError:
            max_steps = [max_step * abs(p.data.real) for p in params]

        defaults = dict(max_steps=max_steps)
        super(CustomBFGSREALOptimizer, self).__init__(params, defaults=defaults)

        self.hesse = torch.eye(len(params), dtype=torch.float64).expand(len(params[0]), -1, -1)

        # store the last parameters in case the simulation becomes unstable
        self.last_params = [p.data.clone() for p in params]
        self.last_grads = None
        self.first_step = True
        self.last_sk = None

        self.max_step = max_step
        self.decay = decay

        self.verbose = verbose

    def decrease_step_size(self):
        self.max_step *= self.decay
        if self.verbose:
            print('Decreasing max. step size to {}'.format(self.max_step))


    def step(self):

        # 1.: -H * nabla
        grads = torch.stack([p.grad.real for p in self.param_groups[0]['params']], dim=1)

        direction_p = torch.matmul(-self.hesse, grads)

        # 2. determine step size to avoid line search
        min_steps = torch.stack([self.max_step*p.data.real / direction_p[:, idx] for idx, p in enumerate(self.param_groups[0]['params'])])
        if self.first_step:
            # ensure sufficiently large first step
            self.first_step = False
            a = torch.min(torch.abs(min_steps), dim=0).values
        else:
            a = torch.min(torch.tensor(1), torch.min(torch.abs(min_steps), dim=0).values)

        # 3. get sk and update parameters
        sk = a.unsqueeze(-1) * direction_p
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                p.data += sk[:, p_idx]

        # 4. get yk to update Hesse
        if self.last_grads is None:
            # Initial gradients
            self.last_grads = grads
            self.last_sk = sk
        else:
            yk = (grads - self.last_grads)

            # dims yk and sk (3, 1, 1000) -> (1000, 3, 1)

            ykT = torch.transpose(yk, -1, -2)
            last_skT = torch.transpose(self.last_sk, -1, -2)

            rho = 1 / torch.matmul(ykT, self.last_sk)

            # 5. update Hesse
            left = torch.eye(len(self.param_groups[0]['params']), dtype=torch.float64).expand(len(self.param_groups[0]['params'][0]), -1, -1) - rho * torch.matmul(self.last_sk, ykT)
            right = torch.eye(len(self.param_groups[0]['params']), dtype=torch.float64).expand(len(self.param_groups[0]['params'][0]), -1, -1) - rho * torch.matmul(yk, last_skT)
            end = rho * torch.matmul(self.last_sk, last_skT)

            self.hesse = torch.matmul(left, torch.matmul(self.hesse, right)) + end

            self.last_grads = grads
            self.last_sk = sk
