#
# utils.py
#

from torch.autograd.functional import jacobian
import torch
import numpy as np
from numpy import pi
import timeit
import random
import sys
try:
    import resource
except ImportError:
    resource = None

from contextlib import contextmanager


from scipy.interpolate import UnivariateSpline

from pykalman import KalmanFilter
from scipy.linalg import expm

from tqdm import tqdm



def _check_param_device(param, old_param_device):
    """This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

def qqdot_to_q(qqdot, dt):
    """
    Convert a timeseries of [q,qdot]_t to a T+1 timeseires of [q]_t
    Args:
        qqdot (torch.tensor): (B,T,qdim*2) [q,qdot]
        dt (float): timestemp
    Returns:
        q (torch.tensor): (B,T+1,qdim) 
    Notes:
        Finds q_{1:T+1} s.t qqdot_t approx [0.5*(q_t+1 + q_t),
                                            (1./dt)*(q_t+1 - q_t)]
    """

    B,T,qdim2 = qqdot.shape
    qdim = qdim2//2

    # find a matrix D that maps q to qddot
    a = torch.randn(T+1,qdim).reshape(-1)
    a.requires_grad = True
    def lam(a):
        a_ = a.reshape(T+1,qdim)
        b1 = 0.5*(a_[1:] + a_[:-1])
        b2  = (1./dt) * (a_[1:] - a_[:-1])
        b = torch.cat([b1,b2],dim=-1)
        return b.reshape(-1)
    D = jacobian(lam, a)

    # solve least squares problem
    q = torch.zeros(B,(T+1)*qdim)
    qqdot = qqdot.reshape(B,-1,1)
    for b in range(B):
        sol = torch.lstsq(qqdot[b], D)[0]
        q[b,:] = sol[:(T+1)*qdim].reshape(-1)

    return q.reshape(B,T+1,qdim)

def bfill_uppertriangle(A, vec):
	ii, jj = np.triu_indices(A.size(-2), k=1, m=A.size(-1))
	A[..., ii, jj] = vec
	return A

def bfill_lowertriangle(A, vec):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A, vec):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A

def parameter_grads_to_vector(parameters):
    """Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        if param.grad is not None:
            vec.append(param.grad.view(-1))
        else:
            zrs = torch.zeros_like(param)
            vec.append(zrs.view(-1))
    return torch.cat(vec)


def vector_to_parameter_grads(vec, parameters):
    """Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # ensure that param requires grad
        assert param.requires_grad

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        param.grad.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def smooth_and_diff(x, dt, nders=2, **kwargs):
    """
    Smooth a set of timeseries and differentiate them.
    Args:
        x (torch.tensor): (B,T,n) timeseries
        dt (float): time-step
        nders (int): number of derivatives to take
        **kwargs: arguments to UnivariateSpline
    Returns:
        smooth_series (list of torch.tensor): list of (B,T,n) smooth tensors (xsm, dxsmdt, ...)
    """

    retvals = [torch.zeros_like(x) for i in range(nders+1)]

    B, T, N = x.shape
    t = np.arange(T) * dt
    for b in range(B):
        for n in range(N):
            ser = x[b,:,n].detach().numpy()
            spl = UnivariateSpline(t, ser, **kwargs)

            retvals[0][b,:,n] = torch.tensor(spl(t))
            for d in range(nders):
                retvals[d+1][b,:,n] = torch.tensor(spl.derivative(d)(t))

    return retvals


def kalman_smooth_and_diff(x, dt, nders=2, em_Q=False):
    """
    Smooth a set of timeseries and differentiate them.
    Args:
        x (torch.tensor): (B,T,n) timeseries
        dt (float): time-step
        nders (int): number of derivatives to take
        em_Q (bool): learn transition cov?
    Returns:
        smooth_series (list of torch.tensor): list of (B,T,n) smooth tensors (xsm, dxsmdt, ...)
    """

    retvals = [torch.zeros_like(x) for i in range(nders+1)]

    em_vars = ['initial_state_mean', 'initial_state_covariance', 
                        'observation_covariance']
    if em_Q:
        em_vars += ['transition_covariance']

    if nders != 2:
        raise NotImplementedError

    A = np.array([[0.,1.,0.],
                  [0.,0.,1.],
                  [0.,0.,0.]])
    Ad = expm(A * dt)
    Bd = np.array([[1.,0.,0.]])

    Q = np.diag([0.001,0.001,1.0])

    B, T, N = x.shape
    for b in tqdm(range(B)):
        for n in range(N):
            ser = x[b,:,n:n+1].detach().numpy()

            kf = KalmanFilter(transition_matrices=Ad, observation_matrices=Bd, transition_covariance=Q)
            kf.em(ser, em_vars=em_vars)
            sm_x, _ = kf.smooth(ser)
            sm_x = torch.tensor(sm_x)

            for i in range(3):
                retvals[i][b,:,n] = sm_x[:,i]

    return retvals
