#
# lagcrit.py
#

import torch
from torch.autograd.functional import jacobian
from ceem.opt_criteria import SOSCriterion
from ceem.utils import temp_require_grad
import scipy.sparse as sp
import numpy as np

from .lagsys import BasicLagrangianSystem

class DELCriterion(SOSCriterion):
    """
    Discrete Euler-Lagrange Criterion
    """

    def __init__(self, t, sqrtlam=1.0, u=None):

        if u is not None:
            raise NotImplementedError

        self._t = t
        self._u = u
        self._sqrtlam = sqrtlam

    def apply_inds(self, q, inds):
        if inds is not None:
            t = self._t[inds]
            q = q[inds]
            u = self._u[inds] if self._u is not None else None
        else:
            t = self._t
            u = self._u
        return t, q, u

    def residuals(self, model, q, inds=None, flatten=True):

        with temp_require_grad([q]):
            with torch.enable_grad():
                t, q_, u = self.apply_inds(q, inds)

                qtm1 = q_[:,:-2]
                qt = q_[:,1:-1]
                qtp1 = q_[:,2:]

                resid = self._sqrtlam * model.discrete_euler_lagrange(qtm1, qt, qtp1)

        return resid.view(-1) if flatten else resid

    def jac_resid_x(self, model, q, sparse=False, sparse_format=sp.csr_matrix, inds=None):

        #q.requires_grad = True # make this safer
        with temp_require_grad([q]):
            jac_dyn_q_ = jacobian(lambda q_: self.residuals(model, q_, inds=inds, flatten=True), q)

        if sparse:
            jac_dyn_q_ = jac_dyn_q_.reshape(jac_dyn_q_.shape[0], -1)
            return sparse_format(jac_dyn_q_.detach().numpy(), dtype=np.float64)
        else:
            return jac_dyn_q_