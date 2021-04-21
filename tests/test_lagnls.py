import torch

from ceem.opt_criteria import *
from ceem.systems import LorenzAttractor
from ceem.dynamics import *
from ceem.smoother import *
from ceem import utils

from delsmm.lagcrit import DELCriterion
from delsmm.lagsys import BasicLagrangianSystem


def test_smoother():

    utils.set_rng_seed(1)

    torch.set_default_dtype(torch.float64)

    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    dt = 0.04

    sys = LorenzAttractor(sigma, rho, beta, C, dt, method='midpoint')

    B = 1
    T = 200
    xs = [torch.randn(B, 1, 3)]
    for t in range(T - 1):
        xs.append(sys.step(torch.tensor([0.] * B), xs[-1]))

    x = torch.cat(xs, dim=1).detach()
    x.requires_grad = True
    y = x.detach()

    sys = BasicLagrangianSystem(qdim=3, dt=dt)

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())
    
    x0 = torch.zeros_like(x)
    x0.requires_grad = True

    obscrit = GaussianObservationCriterion(torch.ones(3), t, y)

    dyncrit = DELCriterion(t)

    # Test GroupSOSCriterion
    crit = GroupSOSCriterion([obscrit, dyncrit])

    xsm, metrics = NLSsmoother(x0, crit, sys, solver_kwargs={'verbose': 2, 'tr_rho': 0., 'max_nfev': 2})

    print('Passed.')

    # Test BlockSparseGroupSOSCriterion
    # crit = BlockSparseGroupSOSCriterion([obscrit, dyncrit])

    # xsm, metrics = NLSsmoother(torch.zeros_like(x), crit, sys)

    print('Passed.')

if __name__ == '__main__':
    test_smoother()
