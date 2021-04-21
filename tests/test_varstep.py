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

	sys = LagrangianDoublePendulum(0.05, 1.,1.,1.,1.,10.)

	q1 = torch.rand(5,1,2) * 2 * np.pi - np.pi
	q2 = q1.clone()

	qs = [q1, q2]

	for t in tqdm(range(200)):
		qt = qs[-2].detach()
		qtp1 = qs[-1].detach()

		nq = sys.variational_step(qt,qtp1)
		qs.append(nq)

	qs = torch.cat(qs, dim=1)

	# for i in range(5):
	# 	plt.subplot(5,1,i+1)
	# 	plt.plot(qs[i])
	# plt.show()

if __name__ == '__main__':
	test()