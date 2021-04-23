#
# test_smooth_and_diff.py
#

import torch
from torch import nn
import matplotlib.pyplot as plt
from ceem.opt_criteria import *
from delsmm.lagcrit import DELCriterion
from delsmm.lagsys import BasicLagrangianSystem
from delsmm.systems.lag_doublepen import LagrangianDoublePendulum
import os
from time import time
import numpy as np

from delsmm.utils import kalman_smooth_and_diff

import pytest

opj = os.path.join

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()

@pytest.mark.skipif(not os.path.exists('./datasets/dubpen_qddot.td'), 
                    reason="requires data generation")
def test():

    torch.manual_seed(1)

    # load the data

    data = torch.load('./datasets/dubpen_qddot.td')
    data_ = torch.load('./datasets/dubpen_0p05.td')
    dt = 0.05

    train_data_ = data_[:8]
    test_data_ = data_[8:16]
    train_data = data[:8]
    test_data = data[8:16]

    t_, q_, y = train_data_[:]
    ttest_, qtest_, ytest = test_data_[:]
    t, q, dq, ddq = train_data[:]
    ttest, qtest, dqtest, ddqtest = test_data[:]

    B, T, qdim = q.shape

    with torch.no_grad():
        std = 0.05 * 10
        smoothed_q, smoothed_dq, smoothed_ddq = kalman_smooth_and_diff(y,dt,2,k=4,s=std)
        if False:
            for b in range(3):
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

if __name__ == '__main__':
    test()

