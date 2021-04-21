#
# test_qddot_train.py
#

from delsmm.utils import qqdot_to_q
from delsmm.smm import StructuredMechanicalModel
from delsmm.systems.lag_doublepen import LagrangianDoublePendulum
import torch
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt


from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp

def test(sys='ldp'):
	torch.set_default_dtype(torch.float64)
	torch.manual_seed(1)

	if sys == 'ldp':
		sys = LagrangianDoublePendulum(0.05, 1.,1.,1.,1.,10., method='rk4')
	else:
		sys = StructuredMechanicalModel(qdim=2, dt=0.05, method='midpoint')

	# generate some random data and compute the system accelerations
	q1 = torch.rand(20,1,2) * 2 * np.pi - np.pi
	q2 = q1.clone() + torch.randn_like(q1) * 0.01

	dt = 0.05

	q = 0.5 * (q1 + q2)
	qdot = (q2-q1)/0.05
	
	qddot = sys.compute_qddot(q,qdot).detach()


	params = ptv(sys.parameters())
	params = params + 1e-1 * torch.randn_like(params)
	vtp(params, sys.parameters())

	opt = torch.optim.Adam(sys.parameters(), lr=1e-4)

	for epoch in range(10000):

		opt.zero_grad()

		qddot_ = sys.compute_qddot(q,qdot, create_graph=True)

		loss = torch.nn.functional.mse_loss(qddot_, qddot)

		loss.backward()

		opt.step()

		if epoch == 0:
			start_loss = float(loss)


		if epoch % 100 == 0:
			print(epoch, float(loss), loss / start_loss)

		if loss / start_loss < 1e-5:
			break

	print(epoch, float(loss), loss / start_loss)

	assert loss / start_loss < 1e-5

if __name__ == '__main__':
	test()
