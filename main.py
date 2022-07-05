#a ce stade je fais n'importe quoi niveau variables locales/globales
from collections import deque
from tqdm import tqdm
import torch
import torchvision
from torch import nn

from autoencoder import Autoencoder

def measure():# this is a simulator function, which basically returns an array of the image observed
    return None

def loss_function(x, y, a_matrix):# the loss, which needs to access to the current model, the estimated matrix A, and the last T measures
    # actually in the article's loss expression, there is a i...
    x_estimate = model.forward(x)    
    autoencoder_loss = torch.linalg.norm(x - x_estimate)
    
    y_encode = model.encode(y)
    
    az = torch.linalg.matrix_power(Aa_matrix, t, *, out=None)@z

def batch_eye(n, bs, device=None, dtype=torch.double):
    x = torch.eye(n, dtype=dtype, device=device)
    x = x.reshape((1, n, n))
    y = x.repeat(bs, 1, 1)

    return y

def learn_ab(X, rho=1., U, max_it=10, h=2, a_dtype='double'):# adapted from Carpentier's code
    bs, n, m = X.shape
    n_aug = h * n + d  # Dimension of augmented state
    # Build Y_h and Yh matrices
    Y_h, Yh = build_shifted_matrices(Y=X, h=h)
    Y_h_aug = torch.zeros((bs, n_aug, m - h), dtype=torch.double, device=X.device)
    Y_h_aug[:, :h * n, :] = Y_h
    Y_h_aug[:, h * n:, :] = U[:, :, h:m]
    Y_h = Y_h_aug
    
    
    A_t_s = []
    # Initialize A with identity, B with zeros, but should rather take the last estimates we have
    A0 = torch.zeros((bs, n, n_aug), dtype=torch.double, device=X.device)
    for i in range(h):
        A0[:, :, i * n:(i + 1) * n] = batch_eye(bs=bs, n=n, dtype=torch.double, device=X.device)
    A0[:, :, -d:] = torch.zeros((bs, n, d), dtype=torch.double, device=X.device)  # Initialize B component
    Yh = X[:, :, h:]
    A_t_s.append(A0.transpose(1, 2))
    rho = rho
    P_b = Y_h.bmm(Y_h.transpose(1, 2)) + rho * batch_eye(bs=bs, n=n_aug, dtype=torch.double,
                                                            device=X.device)
    L_b = torch.cholesky(P_b)
    for k in range(max_it):
        A_ = A_t_s[-1] + torch.cholesky_solve(Y_h.bmm(Yh.transpose(1, 2) - Y_h.transpose(1, 2).bmm(A_t_s[-1])), L_b)
        A_t_s.append(A_)
    A = A_t_s[-1].transpose(1, 2)

    if a_dtype == 'float':
        return A.float()
    elif a_dtype == 'double':
        return A

def mp_inverse(matrix):# the Moore-Penrose inverse of the matrix
    return torch.linalg.pinv(matrix)

def control_system(control_command):# this is a simulator function, which basically apply the given command to the system
    return None

def next_funct(goal_z, current_z):
    vlambda = 1.# arbitraire
    return (1 - vlambda)*current_z + vlambda*goal_z


learning_rate = 1e-4
model = autoencoder().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)# not sure it's ok

def command_system_onestep(goal, current_mes, N_k, m, vlambda):
    d_k = measure()
    current_mes.append(d_k)
    for iter in tqdm(range(N_k)):
        mes_embedding = model.encode(current_mes)
        learning_embed_mes = mes_embedding[:m]
        a_matrix, b_matrix = learn_ab(learning_embed_mes)
        image_mes = model.decode(mes_embedding)
        loss = loss_function(image_mes, current_mes, a_matrix)  #surely other parameters
        optimizer.zero_grad()
        loss.backward()# wrong use of loss, not defined correctly
        optimizer.step()
    a_matrix, b_matrix = learn_ab(learning_embed_mes) # or maybe with mes_embedding...
    current_z = model.encode(d_k)
    goal_z = model.encode(goal)
    next_z = next_funct(goal_z, current_z)
    b_inverse = mp_inverse(b_matrix)
    control_command = b_inverse @ (next_z - a_matrix@current_z)
    
    return current_mes, control_command

def command_system(goal, goal_error, T, N_k_list, m):# strange use of the N_k, here the list is supposed to be of length superior to an unkown value, maybe rather do a function
    current_mes = deque([], maxlen = T)
    error = 2*goal_error
    time = 0
    while error > goal_error:
        time += 1
        current_mes, control_command = command_system_onestep(goal, current_mes, N_k_list[time], m)
        control_system(control_command)
    return error, time