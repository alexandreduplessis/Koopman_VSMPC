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