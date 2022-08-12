import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# tb = SummaryWriter()
class Autoencoder_Dense(nn.Module):
    def __init__(self, dim_obs, dim_embed):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_obs, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_embed)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_embed),
            nn.ReLU(inplace=True),
            nn.Linear(dim_embed, dim_obs)
        )


    def forward(self, x):
        encoded = self.encoded(x)
        decoded = self.decoded(encoded)
        return decoded
    
    def encoded(self, x):
        return self.encoder(x)
    
    def decoded(self, x):
        return self.decoder(x)
    
    def reset_model(self):
        for layers in self.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    print("reset_ok")
                    layer.reset_parameters()
                else:
                    print("ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# max_iter = 500
# data_size = 1000
# model = Autoencoder_Dense()
# x = torch.rand(data_size, 1)
# x = x*10
# y = x*x
# def my_loss(x, y):
#     return torch.mean((y-x)**2)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# mse_loss = torch.nn.MSELoss()
# for i in range(max_iter):
#     for j in range(0, data_size, 10):
#         x_estimate_slice = model(x[j:j+10])
#         loss = mse_loss(x_estimate_slice, y[j:j+10])
#         print(loss.item())
#         tb.add_scalar("Loss", loss, i*10+j)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     tb.add_scalar("Estimate", model(torch.tensor([0.5])), i)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 3),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoded(x)
        decoded = self.decoded(encoded)
        return decoded
    
    def encoded(self, x):
        x = x.unsqueeze(-2)
        return self.encoder(x)
    
    def decoded(self, x):
        return self.decoder(x)