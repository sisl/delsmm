#
# test_smooth_and_diff.py
#

import torch
from torch import nn
import matplotlib.pyplot as plt
from ceem.opt_criteria import *
from ceem import utils
import os
import numpy as np

from delsmm.utils import kalman_smooth_and_diff


from torch.utils.data import TensorDataset

opj = os.path.join

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()

plot = False

def main():

    utils.set_rng_seed(1)

    # load the data    
    for dset in ['0p01', '0p05', '0p10', '0p20', '0p30', '0p40', '0p50', '1p0']:
        for damped in ['', 'damped_']:

            data = torch.load('./datasets/%sdubpen_qddot.td'%damped)
            data_ = torch.load('./datasets/%sdubpen_%s.td'%(damped,dset))
            dt = 0.05

            t_, q_, y = data_[:]
            t, q, dq, ddq = data[:]

            B, T, qdim = q.shape

            with torch.no_grad():
                std = 0.05 * 10
                smoothed_q, smoothed_dq, smoothed_ddq = kalman_smooth_and_diff(y,dt, em_Q=False) # new!

                if plot:
                    for b in range(B):
                        for n in range(2):
                            plt.subplot(3,2,1+n)
                            plt.plot(smoothed_q[b,:,n])
                            plt.plot(q_[b,:,n], '--')
                            plt.plot(y[b,:,n], alpha=0.5)

                            plt.subplot(3,2,3+n)
                            plt.plot(smoothed_dq[b,:,n])
                            plt.plot(dq[b,:,n], '--')


                            plt.subplot(3,2,5+n)
                            plt.plot(smoothed_ddq[b,:,n])
                            plt.plot(ddq[b,:,n], '--')

                        plt.show()

            dataset = TensorDataset(t_, smoothed_q, smoothed_dq, smoothed_ddq)
            torch.save(dataset, './datasets/%sdubpen_%s_smoothed.td'%(damped, dset))

if __name__ == '__main__':
    main()

