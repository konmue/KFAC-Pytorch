import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import trange

from optimizers.kfac import KFACOptimizer


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
        self.net = nn.Sequential(*[nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1)])

    def forward(self, x):
        out = torch.zeros_like(x[:, :, 0])
        for t in range(x.size(1)):
            out[:, t] = self.net(x[:, t, :]).squeeze()
        return out


# NOTE: no interdependcies across time yet...
batch_size = 16
n_steps = 180
n_features = 1

model = MinimalDHModel()
criterion = nn.MSELoss()
optimizer = KFACOptimizer(model, lr=3e-4, n_model_steps=n_steps - 1, TCov=5, TInv=10)

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


n_optimization_steps = 1000
iterator = trange(n_optimization_steps)
losses = []
for i in iterator:
    # R = x[..., 0].diff(dim=1).squeeze()
    # H = torch.outer(R, R)

    optimizer.zero_grad()

    if optimizer.steps % optimizer.TCov == 0:
        optimizer.acc_stats = True

        x = exponential_brownian_motion(1, n_steps).unsqueeze(-1)
        hedge = model(x[:, :-1])
        # portfolio = torch.sum(hedge * x[..., 0].diff(dim=1), dim=1)
        # loss = criterion(portfolio, y)
        loss_sample = hedge.mean()  # TODO: pseudo-criterion?!
        # loss_sample.backward(retain_graph=True)
        loss_sample.backward()

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

for name, p in model.named_parameters():
    shape = (hedge.size(1), p.numel())
    jacobian_block = torch.zeros(shape)
    for t in range(hedge.size(1)):
        g = torch.autograd.grad(hedge[0, t], p, retain_graph=True)[0].detach()
        g = torch.cat([g_.flatten() for g_ in g])
        jacobian_block[t] = g
    g_block = jacobian_block.T @ H @ jacobian_block
    # TODO: move inverse out of loop; can do batchwise
    g_block_inv = torch.cholesky_inverse(g_block)
