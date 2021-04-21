import torch
from delsmm.lagsys import BasicLagrangianSystem
from delsmm.smm import StructuredMechanicalModel, AltStructuredMechanicalModel, ForcedSMM

def test():

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    # generate some test data
    T = 10
    qdim = 2

    q = torch.randn(5, T, qdim)
    q.requires_grad = True

    qtm1 = q[:,:-2]
    qt = q[:,1:-1]
    qtp1 = q[:,2:]

    # init sys
    sys = StructuredMechanicalModel(qdim=qdim, dt=0.1)

    EL = sys.discrete_euler_lagrange(qtm1,qt,qtp1)

    print(EL.shape)

    q = 0.5*(qtm1 + qt).detach()
    v = (1./0.1) * (qt-qtm1).detach()

    print(sys.compute_qddot(q,v).shape)

    # init sys
    sys = AltStructuredMechanicalModel(qdim=qdim, dt=0.1)

    EL = sys.discrete_euler_lagrange(qtm1,qt,qtp1)

    print(EL.shape)

    q = 0.5*(qtm1 + qt).detach()
    v = (1./0.1) * (qt-qtm1).detach()

    print(sys.compute_qddot(q,v).shape)

    # init sys
    sys = ForcedSMM(qdim=qdim, dt=0.1)

    EL = sys.discrete_euler_lagrange(qtm1,qt,qtp1)

    print(EL.shape)

    q = 0.5*(qtm1 + qt).detach()
    v = (1./0.1) * (qt-qtm1).detach()

    print(sys.compute_qddot(q,v).shape)




if __name__ == '__main__':
    test()