#a ce stade je fais n'importe quoi niveau variables locales/globales
from collections import deque
from tqdm import tqdm
import torch
import torchvision
from torch import nn

def measure():# this is a simulator function, which basically returns an array of the image observed
    return None

def loss_function(x, y, a_matrix):# the loss, which needs to access to the current model, the estimated matrix A, the goal, and the last T measures
    return None

def learn_ab():# this is basically what Carpentier proposes, I can surely find the code somewhere
    return None

def mp_inverse(matrix):# the Moore-Penrose inverse of the matrix
    return matrix

def control_system(control_command):# this is a simulator function, which basically apply the given command to the system
    return None


class autoencoder(nn.Module): #fortement aléatoire
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):# not used
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    next_z = (1 - vlambda)*current_z + vlambda*goal_z
    b_inverse = mp_inverse(b_matrix)
    control_command = b_inverse @ (next_z - a_matrix@current_z)
    
    return current_mes, control_command

def command_system(goal, goal_error, T, N_k_list, m, vlambda):# strange use of the N_k, here the list is supposed to be of length superior to an unkown value, maybe rather do a function
    current_mes = deque([], maxlen = T)
    error = 2*goal_error
    time = 0
    while error > goal_error:
        time += 1
        current_mes, control_command = command_system_onestep(goal, current_mes, N_k_list[time], m, vlambda)
        control_system(control_command)
    return error, time