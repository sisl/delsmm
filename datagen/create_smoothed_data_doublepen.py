#
# test_smooth_and_diff.py
#

import matplotlib

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem import logger
from ceem4ln.lagcrit import DELCriterion
from ceem4ln.lagsys import BasicLagrangianSystem
from ceem4ln.systems.lag_doublepen import LagrangianDoublePendulum
import os
import click
from time import time
import numpy as np

from ceem4ln.utils import kalman_smooth_and_diff


from torch.utils.data import TensorDataset

opj = os.path.join

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()


def main():

    torch.manual_seed(1)

    # load the data
    dset = '0p10'
    # dset='1p0'

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

            if False:
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
