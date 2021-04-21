import torch
from torch import nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
from scipy.optimize import root
from ceem.dynamics import *
from ceem.nn import LNMLP
from ceem.utils import temp_require_grad
from tqdm import tqdm
from delsmm.lagsys import AbstractLagrangianSystem
import delsmm.utils as utils

class AbstractStructuredMechanicalModel(AbstractLagrangianSystem, nn.Module):

    def __init__(self, qdim, dt, hidden_sizes=[32]*3, method='midpoint'):
        """
        Args:
            qdim (int): number of generalized coordinates
            dt (float): time-step
            netparams (dict): parameters of Lagrangian neural network
            method (str): integration method
        """

        AbstractLagrangianSystem.__init__(self, qdim, dt, method)
        nn.Module.__init__(self)

        self._xdim = qdim
        self._udim = None
        self._ydim = qdim

    def kinetic_energy(self, q, v):
        mass_matrix = self._mass_matrix(q)
        kenergy = 0.5 * (v.unsqueeze(-2) @ (mass_matrix @ v.unsqueeze(-1))).squeeze(-1)
        return kenergy

    def potential_energy(self, q):
        pot = self._potential(q)
        return pot

    def ke_hessian(self, q, qdot, create_graph=True):
        """
        Compute Hessian of kinetic energy wrt qdot
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
            create_graph (bool): create graph when computing Hessian
        Returns:
            HKEqdqd (torch.tensor): (*, qdim, qdim) kinetic energy Hessian values
        """
        return self._mass_matrix(q)

class StructuredMechanicalModel(AbstractStructuredMechanicalModel):
    def __init__(self, qdim, dt, hidden_sizes=[32]*3, method='midpoint'):
        super().__init__(qdim, dt, hidden_sizes=hidden_sizes, method=method)
        self._mass_matrix = CholeskyMMNet(qdim, hidden_sizes=hidden_sizes)
        self._potential = PotentialNet(qdim, hidden_sizes=hidden_sizes)

class AltStructuredMechanicalModel(AbstractStructuredMechanicalModel):
    def __init__(self, qdim, dt, hidden_sizes=[32]*3, method='midpoint'):
        super().__init__(qdim, dt, hidden_sizes=hidden_sizes, method=method)
        self._mass_matrix = ConvexMMNet(qdim, hidden_sizes=hidden_sizes)
        self._potential = PotentialNet(qdim, hidden_sizes=hidden_sizes)


class ForcedSMM(StructuredMechanicalModel):
    def __init__(self, qdim, dt, hidden_sizes=[32]*3, method='midpoint'):
        super().__init__(qdim, dt, hidden_sizes=hidden_sizes, method=method)
        self._generalized_force = GeneralizedForceNet(qdim, hidden_sizes=hidden_sizes)

    def generalized_forces(self, q, qdot):
        return self._generalized_force(q, qdot)

class CholeskyMMNet(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes=None, bias=1.0, pos_enforce=lambda x: x):
        self._qdim = qdim
        self._bias = bias
        self._pos_enforce = pos_enforce
        super().__init__()

        embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))

    def forward(self, q):
        dims = list(q.shape)
        dims += [dims[-1]] # [..., qdim, qdim]
        if self._qdim > 1:
            L_params = self.out(self.embed(q))

            L_diag = self._pos_enforce(L_params[..., :self._qdim])
            L_diag = L_diag + self._bias
            L_tril = L_params[..., self._qdim:]
            L = q.new_zeros(*dims)
            L = utils.bfill_lowertriangle(L, L_tril)
            L = utils.bfill_diagonal(L, L_diag)
            M = L @ L.transpose(-2, -1)

        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(-2))

        return M

class ConvexMMNet(torch.nn.Module):
    def __init__(self, qdim, hidden_sizes=None, bias=1.0, pos_enforce=lambda x: x):
        self._qdim = qdim
        self._bias = bias
        self._pos_enforce = pos_enforce
        super().__init__()

        embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], int(qdim * (qdim + 1) / 2))

    def forward(self, q):
        dims = list(q.shape)
        dims += [dims[-1]] # [..., qdim, qdim]
        if self._qdim > 1:
            L_params = self.out(self.embed(q))

            L_diag = self._pos_enforce(L_params[..., :self._qdim])
            L_diag += self._bias
            L_offdiag = L_params[..., self._qdim:]
            M = q.new_zeros(*dims)
            M = utils.bfill_lowertriangle(M, L_offdiag)
            M = utils.bfill_uppertriangle(M, L_offdiag)
            M = utils.bfill_diagonal(M, L_diag)
        else:
            M = self._pos_enforce((self.out(self.embed(q)) + self._bias).unsqueeze(-2))

        return M 

class SharedMMVEmbed(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes=[32]*3):
        self._qdim = qdim
        self._hidden_sizes = hidden_sizes
        super().__init__()
        self._lnet = LNMLP(qdim, hidden_sizes[:-1], hidden_sizes[-1], activation='tanh')

    def forward(self, q):
        embed = self._lnet(q)
        return embed

class PotentialNet(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes=[32]*2):
        self._qdim = qdim
        super().__init__()

        embed = SharedMMVEmbed(qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, q):
        return self.out(self.embed(q))

class GeneralizedForceNet(torch.nn.Module):

    def __init__(self, qdim, hidden_sizes=[32]*2):
        self._qdim = qdim
        super().__init__()

        embed = SharedMMVEmbed(2*qdim, hidden_sizes)

        self.embed = embed
        self.out = torch.nn.Linear(hidden_sizes[-1], qdim)

    def forward(self, q, v):
        inp = torch.cat([q,v], dim=-1)
        return self.out(self.embed(inp))