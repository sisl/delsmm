import torch
from torch.autograd import gradcheck

import autograd.numpy as np
from autograd import grad, jacobian

from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp

import numdifftools as nd

from delsmm.smm import StructuredMechanicalModel

from delsmm.utils import parameter_grads_to_vector

from tqdm import tqdm

def test():

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)


    # generate some test data
    T = 10
    qdim = 2

    q = torch.randn(5, T, qdim)
    
    qdot = torch.randn(5,T, qdim)
    qddot = torch.cat((torch.zeros(5, 1, qdim), (qdot[:, 1:, :] - qdot[:, :-1, :]) / 0.1), dim=1)

    x_ = torch.cat([q,qdot], dim=-1)
    x = x_[:,:-1]
    nx = x_[:,1:]

    sys = StructuredMechanicalModel(qdim=qdim, dt=0.1, hidden_sizes=[2])
    def _loss(qddot_true):
        qddot_pred = sys.compute_qddot(q, qdot, create_graph=True)
        return torch.nn.functional.mse_loss(qddot_true, qddot_pred)

    assert gradcheck(_loss, qddot.requires_grad_())

    print("Testing wrt theta")
    def _ploss(params):
        if params.ndim > 1:
            retval = np.zeros(len(params))
            for i in tqdm(range(len(params))):
                retval[i] = _ploss(params[i])
            return retval
            
        vtp(torch.tensor(params), sys.parameters())
        sys.zero_grad()
        qddot_pred = sys.compute_qddot(q, qdot, create_graph=True)
        return torch.nn.functional.mse_loss(qddot, qddot_pred).detach().numpy()


    params0 = ptv(sys.parameters()).detach()

    # gradient = grad(_ploss)(params0.numpy())
    gradient = nd.Gradient(_ploss)(params0.numpy())

    vtp(params0, sys.parameters())
    sys.zero_grad()

    qddot_pred = sys.compute_qddot(q, qdot, create_graph=True)

    loss = torch.nn.functional.mse_loss(qddot, qddot_pred)

    loss.backward()

    gradient_ = parameter_grads_to_vector(sys.parameters()).detach().numpy()

    print(gradient)

    print(gradient_)

    assert np.allclose(gradient_, gradient)


    print("Testing wrt theta")
    def _ploss(params):
        if params.ndim > 1:
            retval = np.zeros(len(params))
            for i in tqdm(range(len(params))):
                retval[i] = _ploss(params[i])
            return retval
            
        vtp(torch.tensor(params), sys.parameters())
        sys.zero_grad()
        nx_pred = sys.step(torch.ones(5,9), x)
        return torch.nn.functional.mse_loss(nx_pred[...,qdim:], nx[...,qdim:]).detach().numpy()


    params0 = ptv(sys.parameters()).detach()

    # gradient = grad(_ploss)(params0.numpy())
    gradient = nd.Gradient(_ploss)(params0.numpy())

    vtp(params0, sys.parameters())
    sys.zero_grad()

    nx_pred = sys.step(torch.ones(5,9), x)
    loss = torch.nn.functional.mse_loss(nx_pred[...,qdim:], nx[...,qdim:])

    loss.backward()

    gradient_ = parameter_grads_to_vector(sys.parameters()).detach().numpy()

    print(gradient)

    print(gradient_)

    assert np.allclose(gradient_, gradient)





    # assert gradcheck(_ploss, ptv(sys.parameters()))


if __name__ == "__main__":
    test()