import torch
from torch import nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
from scipy.optimize import root
from ceem.dynamics import *
from ceem.nn import LNMLP
from ceem.utils import temp_require_grad
from tqdm import tqdm

class AbstractLagrangianSystem(C2DSystem, nn.Module, AnalyticObsJacMixin, DynJacMixin):

    def __init__(self, qdim, dt, method='midpoint'):
        """
        Args:
            qdim (int): number of generalized coordinates
            dt (float): time-step
            method (str): integration method
        """

        C2DSystem.__init__(self, dt=dt, method=method)

        self._qdim = qdim
        self._xdim = qdim
        self._udim = None
        self._ydim = qdim

    def kinetic_energy(self, q, qdot):
        """
        Compute Kinetic Energy
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
        Returns:
            KE (torch.tensor): (*, 1) kinetic energy values
        """
        raise NotImplementedError

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
        qdims = q.shape
        Hdims = list(qdims) + [self._qdim]
        q = q.reshape(-1, self._qdim)
        qdot = qdot.reshape(-1, self._qdim)

        
        with torch.enable_grad():
            with temp_require_grad([q,qdot]):

                def lamfun(qdot_):
                    KE = self.kinetic_energy(q, qdot_) # (*, 1)
                    JKEq = grad(KE.sum(), [qdot_], create_graph=True)[0] # (*, qdim)
                    return JKEq.sum(0)

                HKEqdqd = jacobian(lamfun, qdot, create_graph=create_graph).transpose(0,1)

                HKEqdqd = HKEqdqd.reshape(*Hdims)

                return HKEqdqd

    def potential_energy(self, q):
        """
        Compute Potential Energy
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
        Returns:
            PE (torch.tensor): (*, 1) potential energy values
        """
        raise NotImplementedError

    def generalized_forces(self, q, qdot):
        """
        Compute Generalized Forces
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
        Returns:
            F (torch.tensor): (*, qdim) generalized forces
        """
        return 0. * q


    def lagrangian(self, q, qdot):
        """
        Compute Lagrangian
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
        Returns:
            lag (torch.tensor): (*, 1) Lagrangian values
        """
        return self.kinetic_energy(q, qdot) - self.potential_energy(q)

    def discrete_lagrangian(self, qt, qtp1):
        """
        Compute discrete Lagrangian
        Args:
            qt (torch.tensor): (*, qdim) generalized coordinates at t
            qtp1 (torch.tensor): (*, qdim) generalized coordinates at t+1
        Returns:
            dlag (torch.tensor): (*, 1) discrete Lagrangian values
        """

        q = 0.5*(qt+qtp1)
        qdot = (qtp1 - qt) / self._dt
        return self.lagrangian(q, qdot)

    def discrete_generalized_forces(self, qt, qtp1):
        """
        Compute Generalized Forces
        Args:
            qt (torch.tensor): (*, qdim) generalized coordinates at t
            qtp1 (torch.tensor): (*, qdim) generalized coordinates at t+1
        Returns:
            Fd (torch.tensor): (*, qdim) discrete generalized forces
        """
        q = 0.5*(qt+qtp1)
        qdot = (qtp1 - qt) / self._dt
        return self.generalized_forces(q, qdot)

    def discrete_euler_lagrange(self, qtm1, qt, qtp1, create_graph=True):
        """
        Compute discrete Euler-Lagrange residuals
        Args:
            qtm1 (torch.tensor): (*, qdim) generalized coordinates at t-1
            qt (torch.tensor): (*, qdim) generalized coordinates at t
            qtp1 (torch.tensor): (*, qdim) generalized coordinates at t+1
            create_graph (bool): create graph when taking grad?
        Returns:
            EL (torch.tensor): (B, qdim) Euler Lagrange Residuals
        """

        lag1 = self.discrete_lagrangian(qtm1, qt)
        lag2 = self.discrete_lagrangian(qt, qtp1)

        Fd1 = self.discrete_generalized_forces(qtm1, qt)
        Fd2 = self.discrete_generalized_forces(qt, qtp1)

        Dlag = grad((lag1+lag2).sum(), [qt], create_graph=create_graph)[0]

        avgFd = 0.5 * (Fd1 + Fd2)

        return Dlag + avgFd

    def discrete_euler_lagrange_slow(self, qtm1, qt, qtp1):
        """
        Compute discrete Euler-Lagrange residuals, slow, for checking.
        Args:
            qtm1 (torch.tensor): (B, qdim) generalized coordinates at t-1
            qt (torch.tensor): (B, qdim) generalized coordinates at t
            qtp1 (torch.tensor): (B, qdim) generalized coordinates at t+1
        Returns:
            EL (torch.tensor): (B, qdim) Euler Lagrange Residuals
        """

        B, qdim = qt.shape

        lag1 = self.discrete_lagrangian(qtm1, qt)
        lag2 = self.discrete_lagrangian(qt, qtp1)

        Fd1 = self.discrete_generalized_forces(qtm1, qt)
        Fd2 = self.discrete_generalized_forces(qt, qtp1)

        EL = torch.zeros(B, qdim)

        for b in range(B):
            D1b = grad(lag1[b], [qt], retain_graph=True)[0][b]
            D2b = grad(lag2[b], [qt], retain_graph=True)[0][b]
            EL[b] = D1b + D2b

        return EL + 0.5 * (Fd1 + Fd2)

    def observe(self, t, x, u):
        return x

    def jac_obs_x(self, t, x, u):

        B, T, n = x.shape

        return torch.eye(self.xdim).expand(B, T, n, n)

    def variational_step(self, q1, q2, oneatatime=False):
        """
        Integrate the system forward one step using variational integration.
        Args:
            q1 (torch.tensor): (*, qdim) state at t
            q2 (torch.tensor): (*, qdim) state at t+1
            oneatatime (bool): if True, loop over each (q1[i,:], q2[i,:]) separately
        Returns:
            q3 (torch.tensor): (*, qdim) state at t+2
        """

        dims = q1.shape

        if oneatatime:

            q1 = q1.reshape(-1,self._qdim)
            q2 = q2.reshape(-1,self._qdim)

            q3_ = torch.zeros_like(q1)

            for i in tqdm(range(q1.shape[0])):
                q3_[i] = self.variational_step(q1[i:i+1], q2[i:i+1])

            return q3_.reshape(*dims)


        def closure(q3):
            q3 = torch.tensor(q3).reshape(*dims)

            with temp_require_grad([q2]):

                resid = self.discrete_euler_lagrange(q1,q2,q3)

            return resid.reshape(-1).detach().numpy()

        def jac(q3):

            q3 = torch.tensor(q3).reshape(*dims)

            with temp_require_grad([q2, q3]):

                tmpfun = lambda q_: self.discrete_euler_lagrange(q1,q2,q_).reshape(-1)

                jac = jacobian(tmpfun, q3)
                jac = jac.reshape(jac.shape[0], -1)

            return jac.detach().numpy()

        dq = q2 - q1

        q3_guess = (q2 + dq).detach().numpy().reshape(-1)

        res = root(closure, q3_guess, jac=jac, method='lm')

        assert res.success, res

        q3 = torch.from_numpy(res.x).reshape(*dims)

        return q3

    def compute_qddot(self, q, qdot, create_graph=False):
        """
        Compute qddot from the Euler-Lagrange equation.
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
            create_graph (bool): create graph for diff through qqdot?
        Returns:
            qddot (torch.tensor): (*, qdim) generalized accelerations
        """

        dims = q.shape
        qdim = dims[-1]

        q = q.reshape(-1, qdim)
        qdot = qdot.reshape(-1, qdim)

        F = self.generalized_forces(q, qdot)

        with torch.enable_grad():
            with temp_require_grad([q,qdot]):
                L = self.lagrangian(q, qdot)
                Jq = grad(L.sum(),[q], create_graph=create_graph)[0].unsqueeze(-1)
                Hqdqd = jacobian(lambda qd: grad(self.lagrangian(q, qd).sum(),[qd], create_graph=True)[0].sum(0), 
                    qdot, create_graph=create_graph)
                Hqdqd = Hqdqd.transpose(0,1)
                Hqqd = jacobian(lambda q_: grad(self.lagrangian(q_, qdot).sum(),[qdot], create_graph=True)[0].sum(0), 
                    q, create_graph=create_graph)
                Hqqd = Hqqd.transpose(0,1)

                b = (F.unsqueeze(-1) + Jq - Hqqd @ qdot.unsqueeze(-1))
                qddot = torch.solve(b, Hqdqd)[0].squeeze(-1)

        return qddot.reshape(dims)

    def step_derivs(self, t, x, u=None, create_graph=True):
        """Returns xdot_t

        Args:
            t (torch.tensor): (B, T,) shaped time indices
            x (torch.tensor): (B, T, qdim*2) shaped [q, qdot]
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            xdot (torch.tensor): (B, T, qdim*2) [qdot, qddot]
        """

        q = x[...,:self._qdim]
        qdot = x[...,self._qdim:]

        qddot = self.compute_qddot(q,qdot, create_graph=create_graph)

        return torch.cat([qdot,qddot],dim=-1)







class BasicLagrangianSystem(AbstractLagrangianSystem, nn.Module):

    def __init__(self, qdim, dt, netparams={'hidden_sizes':[32]*3, 'activation':'tanh'}, method='midpoint'):
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

        self._KEnet = LNMLP(input_size=qdim*2, output_size=1, **netparams)
        self._PEnet = LNMLP(input_size=qdim, output_size=1, **netparams)


    def kinetic_energy(self, q, qdot):
        """
        Compute Lagrangian
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
            qdot (torch.tensor): (*, qdim) generalized velocities
        Returns:
            KE (torch.tensor): (*, 1) kinetic energy values
        """
        inp = torch.cat([q,qdot], dim=-1)
        return self._KEnet(inp)


    def potential_energy(self, q):
        """
        Compute Lagrangian
        Args:
            q (torch.tensor): (*, qdim) generalized coordinates
        Returns:
            PE (torch.tensor): (*, 1) potential energy values
        """
        return self._PEnet(q)


