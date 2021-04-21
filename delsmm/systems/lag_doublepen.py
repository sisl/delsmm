#
# lag_doublepen.py
#

from ..lagsys import AbstractLagrangianSystem
import torch
from torch import nn

class LagrangianDoublePendulum(AbstractLagrangianSystem, nn.Module):

    def __init__(self, dt, m1, m2, l1, l2, g, eta = None, method='midpoint'):
        """
        Double Pendulum
        Args:
            dt (float): time-step
            m1 (float): mass 1
            m2 (float): mass 2
            l1 (float): length 1
            l2 (float): length 2
            g (float): gravity
            eta (float): linear joint damping
        """
        AbstractLagrangianSystem.__init__(self, qdim=2, dt=dt, method=method)
        nn.Module.__init__(self)

        self._m1 = m1
        self._m2 = nn.Parameter(torch.tensor([m2]))
        self._l1 = nn.Parameter(torch.tensor([l1]))
        self._l2 = nn.Parameter(torch.tensor([l2]))
        self._eta = eta if eta is None else nn.Parameter(torch.tensor([eta]))
        self._g = g

    @property
    def _params(self):
        return (self._m1, self._m2, self._l1, self._l2, self._g)

    def lagrangian(self, q, qdot):

        t1 = q[...,0:1]
        t2 = q[...,1:2]
        w1 = qdot[...,0:1]
        w2 = qdot[...,1:2]

        m1,m2,l1,l2,g = self._params

        # kinetic energy (T)
        T1 = 0.5 * m1 * (l1 * w1)**2
        T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + \
                        2 * l1 * l2 * w1 * w2 * torch.cos(t1 - t2))
        T = T1 + T2

        # potential energy (V)
        y1 = -l1 * torch.cos(t1)
        y2 = y1 - l2 * torch.cos(t2)
        V = m1 * g * y1 + m2 * g * y2

        return T - V

    def generalized_forces(self, q, qdot):
        """
        Compute Generalized Forces
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
        Returns:
            F (torch.tensor): (*, qdim) generalized forces
        """
        if self._eta is None:
            return 0. * q
        else:
            return -self._eta * qdot

