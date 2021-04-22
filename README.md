[Training Structured Mechanical Models by Minimizing Discrete Euler-Lagrange Residual](https://arxiv.org/abs/xxx) 
=======
Kunal Menda, Jayesh K. Gupta, Zachary Manchester, Mykel J. Kochenderfer

![overall-idea.png](figs/DELSMMOverview.png)

## Getting started
Clone `delsmm` and pip install the module.
```
cd delsmm
pip install -e .
```

## Codebase Overview
- `delsmm`: contains the source code, including the `StructuredMechanicalModel` class, training criteria, and regularization criterion.
- `examples`: contains a Jupyter notebook containing a minimal example of how to smooth data and fit an SMM to it by minimizing the discrete Euler-Lagrange residual.
- `experiments`: contains scripts to run the experiments in the paper and plot the results.
- `datagen`: contains scripts to generate example datasets.


