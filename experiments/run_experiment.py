#
# run_experiment.py
#

"""
Run script for training SMM models on double pendulum data.
"""

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem import logger
from delsmm.lagcrit import DELCriterion
from delsmm.lagsys import BasicLagrangianSystem
from delsmm.smm import StructuredMechanicalModel, ForcedSMM
from delsmm.barriercrit import LogDetBarrierCriterion, MxNormBarrierCriterion
from delsmm.systems.lag_doublepen import LagrangianDoublePendulum
import os
import click
from time import time
import numpy as np

from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp

from torch.utils.data import TensorDataset, DataLoader

opj = os.path.join


def run(seed, lr, method, noise, damped, smoketest=False):
    """
    Args:
        seed (int): seed
        lr (float): init learning rate
        method (str): training method in ['qdd', 'nqqd', 'del+mnorm', 'del+logdet']
        noise (float): amount of noise in [0.01,0.05, 0.1]
        damped (bool): use damped pendulum data
        smoketest (bool): if smoketest, runs 2 epochs
    """
    torch.set_default_dtype(torch.float64)
    dtype=torch.get_default_dtype()

    torch.manual_seed(seed)

    # load the data
    dataset = 'damped_' if damped else ''
    noisedict = {0.01:'0p01', 0.05:'0p05', 0.10: '0p10', 0.20: '0p20', 0.30: '0p30', 0.40: '0p40', 0.50: '0p50', 1.0: '1p0'}
    dataset += 'dubpen_%s_smoothed.td' % noisedict[noise]
    dataset = './datasets/' + dataset

    data = torch.load('./datasets/%sdubpen_qddot.td'%('damped_' if damped else ''))
    data_ = torch.load(dataset)
    dt = 0.05
    logdir = 'data/%s_%s_%.1e_%.3f_%d' % ('damped' if damped else 'undamped', method, lr, noise, seed)
    logger.setup(logdir, action='d')

    inds = torch.randperm(16)

    Btr = 8
    Bte = 4
    Bva = 4


    train_data_ = data_[inds[:Btr]]
    test_data_ = data_[inds[Btr:Btr+Bte]]
    val_data_ = data_[inds[Btr+Bte:Btr+Bte+Bva]]
    train_data = data[inds[:Btr]]
    test_data = data[inds[Btr:Btr+Bte]]
    val_data = data[inds[Btr+Bte:Btr+Bte+Bva]]

    t_, smq, smdq, smddq = train_data_[:]
    ttest_, smqtest, smdqtest, smddqtest = test_data_[:]
    tval_, smqval, smdqval, smddqval = val_data_[:]
    t, q, dq, ddq = train_data[:]
    ttest, qtest, dqtest, ddqtest = test_data[:]
    tval, qval, dqval, ddqval = val_data[:]

    B, T, qdim = q.shape

    # create the appropriate dataloader
    if 'del' in method:
        smq_1 = smq[:,:-2]
        smq_2 = smq[:,1:-1]
        smq_3 = smq[:,2:]
        smq_B = torch.stack([smq_1,smq_2,smq_3], dim=2).reshape(-1,3,2).detach()
        print(smq_B.shape, smq.shape)
        dataset = TensorDataset(smq_B)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    elif method == 'qdd':
        dataset = TensorDataset(smq.reshape(-1,1,2), smdq.reshape(-1,1,2), smddq.reshape(-1,1,2))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    elif method == 'nqqd':
        x = torch.cat([smq, smdq], dim=-1)
        inp = x[:,:-1]
        out = x[:,1:]
        inp = inp.reshape(-1,1,4)
        out = out.reshape(-1,1,4)
        dataset = TensorDataset(inp, out)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    else:
        raise NotImplementedError


    # set up logdir and model
    if damped:
        system = ForcedSMM(qdim=qdim, dt=dt)
    else:
        system = StructuredMechanicalModel(qdim=qdim, dt=dt)

    # create the appropriate closure
    def qddcrit(system, smq_, smdq_, smddq_):

        ddq_ = system.compute_qddot(smq_,smdq_, create_graph=True)
        ddq_loss = torch.nn.functional.mse_loss(ddq_,smddq_)

        return ddq_loss

    def nqqdcrit(system, inp, out):

        out_ = system.step(torch.ones_like(inp)[...,0], inp)
        nqqd_loss = torch.nn.functional.mse_loss(out_,out)

        return nqqd_loss


    if 'del' in method:
        dyncrit = DELCriterion(t_)

        if 'logdet' in method:
            bc = LogDetBarrierCriterion
            bcf = LogDetBarrierCriterion.mineig
        else:
            bc = MxNormBarrierCriterion
            bcf = MxNormBarrierCriterion.mmxnorm

        # initialize the barrier criterion, and find an appropriate coefficient between it and DELcrit
        lb = bcf(system, smq).detach() * 0.99 # interior point init

        barriercrit = bc(lb)
        delcrit = DELCriterion(t)
        with torch.no_grad():
            dyncritloss = dyncrit(system, smq)
            barriercritloss = barriercrit(system, smq)

            mu = float(dyncritloss / barriercritloss) # mu makes them ~equal at init

        barriercrit = bc(lb, mu=mu, x_override=smq)

        crit = GroupCriterion([dyncrit, barriercrit])
    elif method == 'qdd':
        crit = qddcrit
    elif method == 'nqqd':
        crit = nqqdcrit
    else:
        raise NotImplementedError


    # setup optimizer, scheduler
    opt = torch.optim.Adam(system.parameters(), lr = lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 500./(500+k))

    # train
    best_val_loss = np.inf
    best_val_loss_test_qddot = np.inf

    next_params = ptv(system.parameters()).detach()
    
    for epoch in range(2 if smoketest else 500):

        # train with SGD
        for batch in dataloader:

            prev_params = next_params

            opt.zero_grad()

            loss = crit(system, *batch)

            loss.backward()

            opt.step()

            if 'del' in method:
                # check line search
                n_ls  = 0

                while True:

                    next_params = ptv(system.parameters()).detach()

                    del_params = next_params - prev_params

                    with torch.no_grad():
                        c = crit(system, smq)

                    if torch.isnan(c):
                        next_params = prev_params + 0.5 * del_params
                        vtp(next_params, system.parameters())

                        n_ls += 1
                    else:
                        break

        sched.step()

        with torch.no_grad():
            val_sqmddqloss = qddcrit(system, smqval, smdqval, smddqval)
            train_qdd_loss = qddcrit(system, q, dq, ddq)
            test_qdd_loss = qddcrit(system, qtest, dqtest, ddqtest)
            val_qdd_loss = qddcrit(system, qval, dqval, ddqval)

        # select best model using validation error
        if val_sqmddqloss < best_val_loss:
            best_val_loss = float(val_sqmddqloss)
            best_val_loss_test_qddot_loss = float(test_qdd_loss)

            torch.save(system.state_dict(), os.path.join(
                logger.get_dir(), 'ckpts', 'best_model.th'))

        logger.logkv("train/epoch", epoch)
        logger.logkv("train/loss", float(loss))
        logger.logkv("train/log10lr", np.log10(float(opt.param_groups[0]['lr'])))

        logger.logkv("eval/val_sqmddqloss", float(val_sqmddqloss))
        logger.logkv("eval/train_qdd_loss", float(train_qdd_loss))
        logger.logkv("eval/test_qdd_loss", float(test_qdd_loss))
        logger.logkv("eval/val_qdd_loss", float(val_qdd_loss))

        logger.logkv("eval/best_val_loss", float(best_val_loss))
        logger.logkv("eval/best_val_loss_test_qddot_loss", float(best_val_loss_test_qddot_loss))

        logger.dumpkvs()

@click.command()
@click.option('-n', '--noise', type=float, default=0.1, help="Noise setting for the experiment")
@click.option('--parallel', type=int, default=0, help="Set > 1 for number of parallel jobs to run")
def main(noise, parallel):
    methods = ['qdd', 'del+logdet', 'nqqd']
    lrs = [1e-3, 1e-2, 1e-4]
    if parallel > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=parallel)(delayed(run)(seed, lr, method, noise, damped) for method in methods for lr in lrs for damped in [False, True] for seed in range(100))
    else:
        for damped in [False, True]:
            for lr in lrs:
                for seed in range(100):
                    for method in methods: 
                        run(seed, lr, method, noise, damped)


if __name__ == '__main__':
    main()    
