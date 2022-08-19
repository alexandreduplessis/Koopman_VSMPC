# Deep Learning Koopman Control of Dynamical systems for Visual Servoing
_Project by Alexandre DUPLESSIS_

Online control of a robot for visual servoing, based on Koopman theory and an auto-encoder architecture.

The method is tested in a simulated VS environment (see comments) where the robot's camera observes a 2D black square in a white background.

## Prerequisites
- torch
- numpy
- tensorboard
- matplotlib
- gym
- json
- tqdm

## Usage
**Warning:** The code was tested in a devoted conda environment. To make the code work, adapt first line of ```main.py```

To launch code, use ```python main.py```.

To visualize the losses and``, use ```tensorboard --logdir=runs```.

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
|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-seed``|``--seed``         |     None            | Set seed or not               |
|``-rc``|``--random_control``         |     False            | Use backup random control of update it              |


**Display parameters**
|Short |Long               |Default                  |Description                |
|------|------------------|:-------------------------:|---------------------------|
|``-sm``|``--show_model``         |     False            | Show model or not             |     
