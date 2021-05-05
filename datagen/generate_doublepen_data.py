#
# generate_lorenz_data.py
#

import torch
from ceem import utils
from torch.utils.data import TensorDataset
import numpy as np

from delsmm.systems.lag_doublepen import LagrangianDoublePendulum

from tqdm import tqdm
import matplotlib.pyplot as plt

plot = False

def main():

    utils.set_rng_seed(2)

    torch.set_default_dtype(torch.float64)

    dt = 0.05
    sys = LagrangianDoublePendulum(dt, 1.,1.,1.,1.,10.)

    q1 = torch.rand(16,1,2) * np.pi - np.pi/2
    q2 = q1.clone()

    qs = [q1, q2]

    for t in tqdm(range(200)):
        qt = qs[-2].detach()
        qtp1 = qs[-1].detach()

        nq = sys.variational_step(qt,qtp1, oneatatime=True)
        qs.append(nq)

    x = torch.cat(qs, dim=1).detach()

    B, T, _ = x.shape
    t = torch.arange(T).unsqueeze(0).repeat(B,1).float() 

    y_p01 = (x + 0.01 * torch.randn_like(x)).detach()
    y_p05 = (x + 0.05 * torch.randn_like(x)).detach()
    y_p10 = (x + 0.1 * torch.randn_like(x)).detach()
    y_p20 = (x + 0.2 * torch.randn_like(x)).detach()
    y_p30 = (x + 0.3 * torch.randn_like(x)).detach()
    y_p40 = (x + 0.4 * torch.randn_like(x)).detach()
    y_p50 = (x + 0.5 * torch.randn_like(x)).detach()
    y_1p0 = (x + 1.0 * torch.randn_like(x)).detach()

    dataset = TensorDataset(t, x, y_p01)
    torch.save(dataset, './datasets/dubpen_0p01.td')

    dataset = TensorDataset(t, x, y_p05)
    torch.save(dataset, './datasets/dubpen_0p05.td')

    dataset = TensorDataset(t, x, y_p10)
    torch.save(dataset, './datasets/dubpen_0p10.td')

    dataset = TensorDataset(t, x, y_p20)
    torch.save(dataset, './datasets/dubpen_0p20.td')

    dataset = TensorDataset(t, x, y_p30)
    torch.save(dataset, './datasets/dubpen_0p30.td')

    dataset = TensorDataset(t, x, y_p40)
    torch.save(dataset, './datasets/dubpen_0p40.td')

    dataset = TensorDataset(t, x, y_p50)
    torch.save(dataset, './datasets/dubpen_0p50.td')

    dataset = TensorDataset(t, x, y_1p0)
    torch.save(dataset, './datasets/dubpen_1p0.td')


    x_ = (x[:,1:] + x[:,:-1]) * 0.5
    dx = (x[:,1:] - x[:,:-1]) / dt   
    x = x_
    ddx = sys.compute_qddot(x,dx)
    B, T, _ = x.shape
    t = torch.arange(T).unsqueeze(0).repeat(B,1).float() 

    dataset = TensorDataset(t, x, dx,ddx)
    torch.save(dataset, './datasets/dubpen_qddot.td')

    if plot:
        for b in range(16):

            plt.subplot(8,2,b+1)
            plt.plot(x[b])

        plt.show()







if __name__ == '__main__':
    main()