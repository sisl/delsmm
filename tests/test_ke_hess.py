import torch
from delsmm.lagsys import BasicLagrangianSystem, AbstractLagrangianSystem
from delsmm.smm import StructuredMechanicalModel


from delsmm.utils import parameter_grads_to_vector

def test():

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    # generate some test data
    T = 10
    qdim = 2

    q = torch.randn(5, T, qdim)
    
    qdot = torch.randn(5,T, qdim)

    # init sys
    sys = BasicLagrangianSystem(qdim=qdim, dt=0.1)

    mm = sys.ke_hessian(q,qdot)
    print(mm.shape)

    mm.norm().backward()

    print(parameter_grads_to_vector(sys.parameters()).norm())

    sys = StructuredMechanicalModel(qdim=qdim, dt=0.1)
    
    mm = sys.ke_hessian(q,qdot)
    print(mm.shape)

    mm.norm().backward()

    print(parameter_grads_to_vector(sys.parameters()).norm())


    mm = AbstractLagrangianSystem.ke_hessian(sys,q,qdot)
    print(mm.shape)

    mm.norm().backward()

    print(parameter_grads_to_vector(sys.parameters()).norm())



if __name__ == '__main__':
    test()