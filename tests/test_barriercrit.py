#
# test_barriercrit.py
#

from ceem4ln.barriercrit import MxNormBarrierCriterion
from ceem4ln.smm import StructuredMechanicalModel
from ceem4ln.lagcrit import DELCriterion
from ceem.opt_criteria import GroupCriterion
import torch 

def test():

	qdim = 2
	B = 2
	T = 10

	t = torch.arange(T).unsqueeze(0).repeat(B,1).float()
	q = torch.randn(B,T,qdim)

	sys = StructuredMechanicalModel(qdim=qdim, dt=0.1)
	lb = MxNormBarrierCriterion.mmxnorm(sys, q).detach() / 10. # interior point init

	barriercrit = MxNormBarrierCriterion(lb)
	delcrit = DELCriterion(t)

	crit = GroupCriterion([delcrit, barriercrit])

	print(crit(sys, q))



if __name__ == '__main__':
	test()
