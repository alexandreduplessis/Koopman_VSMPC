from src.env.vs_env import VsEnv
from src.env.vs_utils import find_max_size_square_center_coordinates
import numpy as np
import torch

class Last_cumulated(): # m and T are switched here, to modify"""
    def __init__(self, m, h, T, length, width=1):
        self.m = m
        self.h = h
        self.T = T
        self.last_cumulated_list = torch.zeros((m+h+1, length, width))
        # self.last_cumulated_list = torch.tensor([[0.]*dimension]*(m+h+1))
    
    def append(self, x):
        """ Shift all value except the last on the right, and replace first element with x """
        copy = torch.clone(self.last_cumulated_list)
        # self.last_cumulated_list[:-1] = copy[1:]
        # self.last_cumulated_list[-1,:] = x.T
        self.last_cumulated_list[:-1] = copy[1:]
        self.last_cumulated_list[-1,:,:] = x
        return None
    
    def return_list(self):
        """ Return the raw list """
        return self.last_cumulated_list
    
    def before(self, nb=None):
        """ Return the last nb elements of the list, each concatenated with the h previous ones """
        if nb == None:
            nb = self.T
        before_list = torch.stack([torch.cat([torch.cat([self.last_cumulated_list[j+self.h-i,:]]) for i in range(self.h, -1, -1)], 0) for j in range(nb)], 0)
        return before_list
    
    def after(self, nb=None):
        """ Return the last nb elements of the list but shifted of one on the right, each concatenated with the h previous ones """
        if nb == None:
            nb = self.T
        after_list = torch.stack([torch.cat([torch.cat([self.last_cumulated_list[j+self.h-i+1,:]]) for i in range(self.h, -1, -1)], 0) for j in range(nb)], 0)
        return after_list
    
    def __str__(self):
        """ Return a string representation of the object """
        return "{}".format(self.last_cumulated_list)


def lr_function(epoch):
    """ Scheduler custom function"""
    if epoch < 20:
        return 10.
    elif epoch < 60:
        return 5.
    elif epoch < 100:
        return 1.
    else:
        return 0.1 

# Works for 1D LSD
# def lr_function(ep
# #och):
    # if epoch < 10:
    #
        # return 10.
        #
    # elif epoch < 
    # #200:
        # return 1.
        #
    # elif epoch < 4
    # #00:
        # r
        # #eturn 0.5
    # else:
    #
        # return 0.1


def estimate(A, B, z, u, T, m, device, method=None):
    """ Estimate the T-m+1 last states of the modeled system in the latent space (for comparison with the true system)
    method: None or 'from_first'
        -> 'from_first' if we start from the first state and recursively estimate the others
        -> None if each state is estimated from the previous one (prevents from diverging in case the learning is bad)
    """
    if method is None:
        z_estimate = A @ z.T + B @ u.T
    elif method == 'from_first':
        z_estimate = torch.zeros((T-m, z.size()[1])).to(device)
        # print(">>>>", z.size())
        z = z[0,:].unsqueeze(-1)
        # # print(z.size())
        # print(B.size())
        # print((torch.tensor(u[0]).unsqueeze(-1)).size())
        # print("Az : ", ((A @ z.T).size()))
        # print("Bu : ", (B @ torch.tensor(u[0]).unsqueeze(-1).unsqueeze(-1)).size())
        # print("u : ", (torch.tensor(u[0]).unsqueeze(-1).unsqueeze(-1)).size())
        # print(u[0].unsqueeze(-1).unsqueeze(-1))
        print("z : ", z.T.size())
        for t in range(T-m):
            # print("z : ", z.size())
            z = (A @ z) + B @ (torch.tensor(u[t]).unsqueeze(-1))
            # print("z :", z.size())
            # print(t)
            # print("z_esti :", z_estimate.size())
            z_estimate[t,:] = z.T
        # print(">>>>>>>", z_estimate.size())
    return z_estimate.T
    
    return z_estimate

    



def find_AB(Z1, Z2, U1, c, n, T, rho, max_iter, device, tensorboard, abscisse):
    """Find A and B approximations iteratively (with control)

    Args:
        Z1 (_type_): T embed measurements
        Z2 (_type_): T next embed measurements
        U1 (_type_): T control entries
        c (int): dimension of control space
        n (int): dimension of latent space
        T (int): number of measurements the approximations are based on
        rho (float): regularity constant
        max_iter (int, optional): number of iterations. Defaults to 10.

    Returns:
        _type_: approximations of A and B
    """
    Z1 = Z1.T
    Z2 = Z2.T
    U1 = U1.T.squeeze()
    first_line = torch.cat(
        [
            torch.eye(n).to(device),
            torch.zeros((n, c)).to(device),
            Z1
        ], 
        dim=1
    )
    
    second_line = torch.cat(
        [
            torch.zeros((c, n)).to(device),
            torch.eye(c).to(device),
            U1
        ], 
        dim=1
    )

    third_line = torch.cat(
        [
            Z1.T,
            U1.T,
            (-rho*torch.eye(T)).to(device)
        ],
        dim=1
    )
    
    omega = torch.cat(
        [
            first_line,
            second_line,
            third_line
        ],
        dim=0
    )
    mat_lambda = torch.zeros((n, T)).to(device)
    theoretical_estimate = torch.linalg.lstsq(torch.cat([Z1.T, U1.T], dim=1), Z2.T).solution
    A_th_estimate = theoretical_estimate[:n, ::].T
    B_th_estimate = theoretical_estimate[n:, ::].T
    theoretical_diff = (Z2 - A_th_estimate @ Z1 - B_th_estimate @ U1).square().mean()
    
    for i in range(max_iter):
        image = torch.cat(
            [
                torch.zeros((n + c, n)).to(device),
                Z2.T - (rho*mat_lambda.T).to(device)
            ],
            dim=0
        )
        new_estimate = torch.linalg.inv(omega) @ image # torch.cholesky_inverse
        A_estimate = new_estimate[:n, ::].T
        B_estimate = new_estimate[n:n+c, ::].T
        mat_lambda = new_estimate[n+c:, ::].T
        new_diff = Z2 - A_estimate @ Z1 - B_estimate @ U1
        performance = new_diff.square().mean()
        tensorboard.add_scalar("Learning A,B", performance, i)
        if performance <  .001:
            break
    # print("Performance of learning of A and B :", performance)
    # print("Theoretical performance of learning of A and B :", theoretical_diff)
    loss_fct = torch.nn.MSELoss()
    linearization_error = loss_fct(new_estimate[:n+c, ::] @ torch.linalg.pinv(new_estimate[:n+c, ::]), torch.eye(n+c).to(device))
    # print("B error :", linearization_error)
    return A_estimate, B_estimate, linearization_error


# def smart_loss(d1, d2):
#     """ Compute smart loss between two VS camera inputs"""
#     abs1, ord1, size1 = find_max_size_square_center_coordinates(d1.cpu().numpy())
#     abs2, ord2, size2 = find_max_size_square_center_coordinates(d2.cpu().numpy())
#     return (abs1 - abs2)**2 + (ord1 - ord2)**2 + (size1 - size2)**2

def auto_loss(model, d):
    """ Compute the auto-encoder loss of the model"""
    loss_fct = torch.nn.MSELoss()
    return loss_fct(d, model(d))
    # return smart_loss(d, model(d))

def pred_loss(model, d, z_estimate, m):
    """ Compute the prediction loss of the model"""
    loss_fct = torch.nn.MSELoss()
    return loss_fct(d[m+1:], model.decoded(torch.tensor(z_estimate))[:-1])
    # return smart_loss(d[m+1:], model.decoded(torch.tensor(z_estimate))[:-1])

def odc_loss(model, d, z_estimate, m):
    """ Compute the total loss of the model"""
    # return auto_loss(model, d, z_estimate, m) + pred_loss(model, d, z_estimate, m)
    return auto_loss(model, d) + pred_loss(model, d, z_estimate, m)