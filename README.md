**Koopman Control of Dynamical systems for Visual Servoing**
Project by Alexandre DUPLESSIS

## Prerequisites
- torch
- numpy
- tensorboard

## Usage
**Warning:** The code was tested in a devoted conda environment. To make the code work, delete first line of ```main.py```
```python main.py```

### Options
Here are the different options available.

**Hyper-parameters**
|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-h``|``--help``         |                         |Show help                  |
|``-e`` |       ``--epochs``            |   1000       |Number of epochs, constant over steps         |
|  ``-s``    |``--steps``      |             5            |Number of training data             |
|  ``-a``    |``--alpha``       |1. |Weight of auto_loss in loss                |
|   ``-b``   |``--beta`` |1.                |Weight of pred_loss in loss             |
|  ``-rho``    |``--regularization``   |1e4|ADAM weight decay              |



**Learning parameters**

|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-T``|``--learning_horizon``         |      1000                   |Horizon of learning                  |
|``-m`` |       ``--AB_horizon``            |   500       |Number of values used to compute A and B         |
|  ``-lr``    |``--lr``      |             1e-3            |ADAM learning rate             |
|  ``-wd``    |``--weight_decay``       |1e-4 |ADAM weight decay                |
|   ``-n``   |``--embed_dim`` |202                |Dimension of latent space             |
|  ``-sh``    |``--secondary_horizon``   |0|Time horizon for dependancy             |

**Environment parameters**
|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-d0``|``--init_pos``         |     random.uniform(-20., 20.)                 | Initial position                  |
|``-dstar`` |       ``--goal_pos``            |  random.uniform(-20., 20.)     | Desired position        |
|  ``-o``    |``--obs_dim``      |          1         | Dimension of observation space          |
|  ``-c``    |``--control_dim``       | 1 |  Dimension of control space              |


**Random parameters**
|------|------------------|-------------------------|---------------------------|
|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-seed``|``--seed``         |     ```None```              | Set seed or not               |
