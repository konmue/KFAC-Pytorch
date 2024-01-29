import matplotlib.pyplot as plt
import torch
from nngeometry.generator import Jacobian
from nngeometry.metrics import FIM
from nngeometry.object import PMatBlockDiag, PMatDiag, PMatKFAC, PVector
from torch import nn
from tqdm import trange

from optimizers.kfac import KFACOptimizer


def compute_block_jacobian_naive(model, func, examples, batch: int = 0):
    hedge = func(examples)
    all_jacobian_blocks = {}
    for name, p in model.named_parameters():
        jacobian_block = torch.zeros(hedge.size(1), p.numel())
        for t in range(hedge.size(1)):
            g = torch.autograd.grad(hedge[batch, t], p, retain_graph=True)[0].detach()
            g = torch.cat([g_.flatten() for g_ in g])
            jacobian_block[t] = g
        all_jacobian_blocks[name] = jacobian_block

        # g_block = jacobian_block.T @ H @ jacobian_block
        # TODO: move inverse out of loop; can do batchwise
        # g_block_inv = torch.cholesky_inverse(g_block)

    return all_jacobian_blocks


def compute_block_jacobian_functional(model, x):
    params = dict(model.named_parameters())

    def func(p, data):
        hedge = torch.func.functional_call(model, p, data)
        return hedge

    # TODO: try vmap again
    jacobian_func = torch.func.jacrev(func, argnums=0)

    jacobian = jacobian_func(params, x)
    return jacobian


def compute_jacobian_nngeometry(model, func, examples, n_output, batch: int = 0):
    generator = Jacobian(model, function=func, n_output=n_output)
    # M_kfac = PMatKFAC(generator=generator, examples=examples)
    # G_kfac = M_kfac.get_dense_tensor(split_weight_bias=True)
    M_blockdiag = PMatBlockDiag(generator=generator, examples=examples)
    G_blockdiag = M_blockdiag.get_dense_tensor()
    return G_blockdiag


def exponential_brownian_motion(batch_size, n_steps, mu=0.0, sigma=0.2, dt=1 / 250):
    log_returns = (
        torch.randn(batch_size, n_steps) * sigma * dt**0.5
        + (mu - sigma**2 / 2) * dt
    )
    log_returns = torch.cat([torch.zeros(batch_size, 1), log_returns], dim=1)
    return torch.exp(log_returns.cumsum(dim=1))


def aa_hook(module, inp):
    a = inp[0].data
    a = a.squeeze() if a.ndim == 2 else a
    module.aa = torch.outer(a, a) if a.ndim == 1 else a**2


class MinimalDHModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            ]
        )

    def forward(self, x):
        out = torch.zeros_like(x[:, :, 0])
        for t in range(x.size(1)):
            out[:, t] = self.net(x[:, t, :]).squeeze()
        return out


# NOTE: no interdependcies across time yet...
batch_size = 16
n_steps = 20
n_features = 1

model = MinimalDHModel()
criterion = nn.MSELoss()

"""
x = exponential_brownian_motion(1, n_steps).unsqueeze(-1)
def jacobian_func(x):
    return model(x[:, :-1])
block_jacobians = compute_block_jacobian_naive(model, jacobian_func, examples=x)
print(block_jacobians.keys())
x = exponential_brownian_motion(3, n_steps).unsqueeze(-1)
func_jacobian = compute_block_jacobian_functional(model, x[:, :-1])
print(func_jacobian.keys())
for k, v in func_jacobian.items():
    print(k, v.shape)
"""

optimizer = KFACOptimizer(
    model, lr=3e-4, n_model_steps=n_steps - 1, TCov=1, TInv=1, stat_decay=0.95
)

"""
for module in model.net.modules():
    if isinstance(module, nn.Linear):
        model.net.register_forward_pre_hook(aa_hook)
"""
# TODO: wait what did I do now? Grad output is now the gradient wrt. the loss function!
# Can I fix this using pseudo-criterion?

# TODO: was there somewhere an assumption regarding the loss function type? Like BCE
# TODO: implement with H_L
# TODO: make more efficient; think about: where to put running average

MEAN_HEDGE_PSEUDO_CRITERION = True

n_optimization_steps = 100
iterator = trange(n_optimization_steps)
losses = []
for i in iterator:
    optimizer.zero_grad()

    if optimizer.steps % optimizer.TCov == 0:
        optimizer.acc_stats = True
        x = exponential_brownian_motion(1, n_steps).unsqueeze(-1)

        R = x[..., 0].diff(dim=1).squeeze()
        H = torch.outer(R, R) + 0.001 * torch.eye(n_steps)
        optimizer.H = H

        hedge = model(x[:, :-1])
        # portfolio = torch.sum(hedge * x[..., 0].diff(dim=1), dim=1)
        # loss = criterion(portfolio, y)
        # TODO: make efficient by not using masked hedges.
        if MEAN_HEDGE_PSEUDO_CRITERION:
            loss_sample = hedge.mean()  # TODO: pseudo-criterion?!
            # loss_sample.backward(retain_graph=True)
            loss_sample.backward()
        else:
            for i in range(hedge.size(1)):
                hedge[0, i].backward(retain_graph=True)

        optimizer.acc_stats = False
        optimizer.zero_grad()

    x = exponential_brownian_motion(batch_size, n_steps).unsqueeze(-1)
    y = torch.relu(x[:, -1, 0] - 1.0)
    hedge = model(x[:, :-1])
    portfolio = torch.sum(hedge * x[..., 0].diff(dim=1), dim=1)
    loss = criterion(portfolio, y)
    loss.backward()
    optimizer.step()

    iterator.set_description(f"Loss: {loss.item():.5f}")
    losses.append(loss.item())

plt.plot(losses)
plt.show()

exit()

for module in model.net.modules():
    if isinstance(module, nn.Linear):
        # model.net.register_forward_pre_hook()
        print(module.aa)
