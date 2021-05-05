#
# test_lagcrit.py
#

import torch
from delsmm.lagsys import BasicLagrangianSystem
from delsmm.lagcrit import DELCriterion

def test():

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    # generate some test data
    T = 10
    qdim = 2

    B = 4

    t = torch.arange(T).unsqueeze(0).repeat(B,1)

    q = torch.randn(B, T, qdim)
    q.requires_grad = True

    qtm1 = q[:-2]
    qt = q[1:-1]
    qtp1 = q[2:]

    # init sys
    sys = BasicLagrangianSystem(qdim=qdim, dt=0.1)

    crit = DELCriterion(t)

    c = crit(sys, q)
    print(c)

    J = crit.jac_resid_x(sys, q)
    print(J)


if __name__ == '__main__':
    test()