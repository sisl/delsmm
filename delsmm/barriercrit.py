#
# barriercrit.py
#

import torch
from ceem.opt_criteria import Criterion
from delsmm.smm import AbstractStructuredMechanicalModel, AbstractLagrangianSystem

class MxNormBarrierCriterion(Criterion):

    def __init__(self, lb, mu=1.0, x_override=None):
        """
        Barrier specifying a minimum value for the expected matrix norm.
        Args:
            lb (float): minimum value
            mu (float): scale coefficient
            x_override (torch.tensor): (B,T,n) system states to use in place of what is passed to forward
        """
        self._lb = lb
        self._mu = mu
        self._x_override = x_override.detach() if x_override is not None else x_override


    def forward(self, model, x, **kwargs):
        """
        Forward method for computing criterion
        Args:
            model (AbstractStructuredMechanicalModel)
            x (torch.tensor): (B,T,n) system states
        Returns:
            criterion (torch.tensor): scalar criterion
        """

        assrcrit = isinstance(model, AbstractLagrangianSystem)
        assert assrcrit, 'model must be an AbstractLagrangianSystem'

        if self._x_override is not None:
            x = self._x_override

        if isinstance(model, AbstractStructuredMechanicalModel):
            mm = model._mass_matrix(x.detach())
        else:
            x_ = 0.5 * (x[:,1:] + x[:,:-1])
            dx_ = (1./model._dt) * (x[:,1:] - x[:,:-1])
            mm = model.ke_hessian(x_.detach(), dx_.detach())
        mmnorm = mm.norm(dim=(-2,-1)).mean()

        return -self._mu * torch.log(mmnorm - self._lb)


    @staticmethod
    def mmxnorm(model, x):
        """
        Compute mean mass-matrix norm
        Args:
            model (AbstractStructuredMechanicalModel)
            x (torch.tensor): (B,T,n) system states
        Returns:
            mmnorm (torch.tensor): scalar mean matrix norm
        """
        assrcrit = isinstance(model, AbstractLagrangianSystem)
        assert assrcrit, 'model must be an AbstractLagrangianSystem'


        if isinstance(model, AbstractStructuredMechanicalModel):
            mm = model._mass_matrix(x.detach())
        else:
            x_ = 0.5 * (x[:,1:] + x[:,:-1])
            dx_ = (1./model._dt) * (x[:,1:] - x[:,:-1])
            mm = model.ke_hessian(x_.detach(), dx_.detach())
        mmnorm = mm.norm(dim=(-2,-1)).mean()

        return mmnorm

class LogDetBarrierCriterion(Criterion):

    def __init__(self, alpha, mu=1.0, x_override=None):
        """
        Barrier specifying a minimum value for the expected matrix norm.
        Args:
            alpha (float): minimum eigenvalue
            mu (float): scale coefficient
            x_override (torch.tensor): (B,T,n) system states to use in place of what is passed to forward
        """
        self._alpha = alpha
        self._mu = mu
        self._x_override = x_override.detach() if x_override is not None else x_override


    def forward(self, model, x, **kwargs):
        """
        Forward method for computing criterion
        Args:
            model (AbstractStructuredMechanicalModel)
            x (torch.tensor): (B,T,n) system states
        Returns:
            criterion (torch.tensor): scalar criterion
        """

        assrcrit = isinstance(model, AbstractLagrangianSystem)
        assert assrcrit, 'model must be an AbstractLagrangianSystem'

        if self._x_override is not None:
            x = self._x_override

        if isinstance(model, AbstractStructuredMechanicalModel):
            mm = model._mass_matrix(x.detach())
        else:
            x_ = 0.5 * (x[:,1:] + x[:,:-1])
            dx_ = (1./model._dt) * (x[:,1:] - x[:,:-1])
            mm = model.ke_hessian(x_.detach(), dx_.detach())
        
        n = x.shape[-1]
        det = (mm - self._alpha * torch.eye(n).reshape(1,1,n,n)).det()

        return -self._mu * torch.log(det).mean()


    @staticmethod
    def mineig(model, x):
        """
        Compute the minimum eigenvalue of the mass-matrix for x
        Args:
            model (AbstractStructuredMechanicalModel)
            x (torch.tensor): (B,T,n) system states
        Returns:
            mmnorm (torch.tensor): scalar mean matrix norm
        """
        assrcrit = isinstance(model, AbstractLagrangianSystem)
        assert assrcrit, 'model must be an AbstractLagrangianSystem'


        if isinstance(model, AbstractStructuredMechanicalModel):
            mm = model._mass_matrix(x.detach())
        else:
            x_ = 0.5 * (x[:,1:] + x[:,:-1])
            dx_ = (1./model._dt) * (x[:,1:] - x[:,:-1])
            mm = model.ke_hessian(x_.detach(), dx_.detach())
        mineig_ = mm.symeig()[0].min()

        return mineig_

