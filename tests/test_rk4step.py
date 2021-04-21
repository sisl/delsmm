# 
# test_varstep.py
#

from delsmm.systems.lag_doublepen import LagrangianDoublePendulum
import torch
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

def test():
	torch.set_default_dtype(torch.float64)
	torch.manual_seed(1)

	sys = LagrangianDoublePendulum(0.05, 1.,1.,1.,1.,10., method='rk4')

	q1 = torch.rand(5,1,2) * 2 * np.pi - np.pi
	q2 = q1.clone() + torch.randn_like(q1) * 0.01

	q = 0.5 * (q1 + q2)
	qdot = (q2-q1)/0.05

	sys.compute_qddot(q,qdot)

	x = torch.cat([q,qdot],dim=-1)

	xs = [x]

	for t in tqdm(range(200)):
		nx = sys.step(torch.zeros(5,1).float(), xs[-1]).detach()
		xs.append(nx)

	xs = torch.cat(xs, dim=1)
	qs = xs[...,:2]

	# for i in range(5):
	# 	plt.subplot(5,1,i+1)
	# 	plt.plot(qs[i])
	# plt.show()

if __name__ == '__main__':
	test()