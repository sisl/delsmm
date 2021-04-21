# 
# test_qqdot_to_q.py
#

from delsmm.utils import qqdot_to_q
from delsmm.systems.lag_doublepen import LagrangianDoublePendulum
import torch
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

def test():
	torch.set_default_dtype(torch.float64)
	torch.manual_seed(1)

	sys = LagrangianDoublePendulum(0.05, 1.,1.,1.,1.,10., method='rk4')

	q1 = torch.rand(2,1,2) * 2 * np.pi - np.pi
	q2 = q1.clone() + torch.randn_like(q1) * 0.01

	dt = 0.05

	q = 0.5 * (q1 + q2)
	qdot = (q2-q1)/0.05

	sys.compute_qddot(q,qdot)

	x = torch.cat([q,qdot],dim=-1)

	xs = [x]

	for t in tqdm(range(50)):
		xs.append(sys.step(torch.zeros(5,1).float(), xs[-1]))

	xs = torch.cat(xs, dim=1)

	B,T,_ = xs.shape

	q = qqdot_to_q(xs, dt)

	# plt.plot(q[0].detach(), '*')
	# plt.plot(np.arange(T) + 0.5,xs[0,:,:2])

	# plt.show()

if __name__ == '__main__':
	test()