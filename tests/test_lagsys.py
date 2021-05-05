import torch
from delsmm.lagsys import BasicLagrangianSystem

def test():

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    # generate some test data
    T = 10
    qdim = 2

    q = torch.randn(T, qdim)
    q.requires_grad = True

    qtm1 = q[:-2]
    qt = q[1:-1]
    qtp1 = q[2:]

    # init sys
    sys = BasicLagrangianSystem(qdim=qdim, dt=0.1)

    EL = sys.discrete_euler_lagrange(qtm1,qt,qtp1)
    ELslow = sys.discrete_euler_lagrange_slow(qtm1,qt,qtp1)

    assert torch.allclose(EL, ELslow)




if __name__ == '__main__':
    test()