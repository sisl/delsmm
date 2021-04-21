#
# plot_exp_results.py
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
opj = os.path.join

def main():

    damped = []
    lr = []
    seed = []
    noise = []
    method = []
    error = []

    noise_ = 0.1
    
    for damped_ in [False, True]:
        for lr_ in [1e-2, 1e-3, 1e-4]:
            for seed_ in range(105):
                for method_ in ['del+logdet', 'qdd', 'nqqd']: 
                    
                    logdir = 'data/%s_%s_%.1e_%.3f_%d' % ('damped' if damped_ else 'undamped', method_, lr_, noise_, seed_)
                    
                    try:
                        df = pd.read_csv(opj(logdir, 'progress.csv'))

                        damped.append('damped' if damped_ else 'undamped')
                        lr.append(lr_)
                        seed.append(seed_)
                        method.append(method_)
                        error.append(df['eval/best_val_loss_test_qddot_loss'].values[-1])
                        noise.append(noise_)
                    except (FileNotFoundError, pd.errors.EmptyDataError):
                        pass

    df = pd.DataFrame(dict(damped=damped, lr=lr, seed=seed, noise=noise, method=method, error=error))

    df = df.loc[df.error < 1e3]

    df_ = df.groupby(['damped', 'noise', 'method', 'lr']).agg(
        mean_error=pd.NamedAgg(column='error', aggfunc=np.mean),
        std_error=pd.NamedAgg(column='error', aggfunc=lambda x: np.std(x) / np.sqrt(len(x))))

    df_mm = df.loc[df.lr==0.0001].groupby('seed').agg(minmax_error=pd.NamedAgg(column='error', aggfunc=np.max))

    df_seed = df_mm.loc[df_mm.minmax_error == float(df_mm.min().values)]

    for damped in ['damped', 'undamped']:

        plt.figure(figsize=(4,3))

        xs = [0,1,2]
        df__ = df_.loc[(damped, noise_, 'del+logdet')]
        plt.bar(xs, df__.mean_error.values, yerr=df__.std_error.values, label='DEL-residual')


        xs = [4,5,6]
        df__ = df_.loc[(damped, noise_, 'qdd')]
        plt.bar(xs, df__.mean_error.values, yerr=df__.std_error.values, label='Acceleration Regression')

        xs = [8,9,10]
        df__ = df_.loc[(damped, noise_, 'nqqd')]
        plt.bar(xs, df__.mean_error.values, yerr=df__.std_error.values, label='Next-state Regression')

        xs = [0,1,2,4,5,6,8,9,10]
        ticks = [r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$'] * 3
        plt.xticks(xs, ticks, rotation=90)

        plt.xlabel(r'Learning Rate')
        plt.ylabel(r'Test Error [ms$^{-2}$]')

        plt.legend(loc='lower left')

        if damped == 'damped':
            plt.title('Damped Pendulum')
        else:
            plt.title('Undamped Pendulum')

        plt.tight_layout()

        plt.savefig('figs/%s_results.pdf'%damped)

        # plt.show()



if __name__ == '__main__':
    main()
