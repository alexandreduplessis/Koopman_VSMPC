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

def mp_inverse(matrix):# the Moore-Penrose inverse of the matrix
    return torch.linalg.pinv(matrix)