import torch
from environment import env_function
from utils import estimate, find_AB, odc_loss, auto_loss, pred_loss

def control_step(current_state, dstar, N, d, u, n, c, m, T, rho, model, optimizer, tb, control_iter, device, alpha, beta, scheduler, secondary_horizon):
    d_before = d.before()
    d_before = d_before.to(device)
    d_after = d.after()
    d_after = d_after.to(device)
    d_before_T = d.before(T)
    d_before_T = d_before_T.to(device)
    u_list = u.return_list()
    u_list = u_list.to(device)
    
    # model.reset_model()
        
    for epoch in range(1, N+1):
        
        
        z_before = model.encoded(d_before)
        z_after = model.encoded(d_after)
        
        A, B, linearization_error = find_AB(Z1=z_before, Z2=z_after, U1=u_list[:m], c=c, n=n, T=m, rho=rho, max_iter=400, device=device, tensorboard=tb, abscisse=(control_iter-1)*N + epoch)
        A = A.to(device)
        B = B.to(device)
        
        z_estimate = estimate(A, B, model.encoded(d_before_T)[m:], u_list[m:], T, m, device)
        
        test = d_before_T[m+1:] - env_function(d_before_T[m:], u_list[m:])[:-1]
        if (test.sum()!=0.).item():
            print("Error :", test.sum())
        
        my_auto_loss = auto_loss(model, d_before_T)
        list_pred_loss = d_before_T[m+1:] - model.decoded(torch.tensor(z_estimate.T))[:-1]
        print("first : {}, last : {}".format(list_pred_loss[0], list_pred_loss[-1]))
        my_pred_loss = pred_loss(model, d_before_T, z_estimate, m)
        loss = alpha*my_auto_loss + beta*my_pred_loss# + linearization_error
        
        tb.add_scalar("Auto_Loss", my_auto_loss, (control_iter-1)*N + epoch)
        tb.add_scalar("Pred_Loss", my_pred_loss, (control_iter-1)*N + epoch)
        tb.add_scalar("Linearization_Error", linearization_error, (control_iter-1)*N + epoch)
        tb.add_scalar("Loss", loss, (control_iter-1)*N + epoch)
        
        with torch.no_grad():
            temp_control =\
                torch.linalg.pinv(B)\
                    @ \
                (
                    model.encoded(torch.cat([dstar]*(secondary_horizon + 1)).to(device)).T\
                        - \
                    A @ model.encoded(d.before(T+1).to(device))[-1].T
                )
            temp_state = env_function(current_state, temp_control)
            tb.add_scalar("Estimation of new error of state", (temp_state - dstar), (control_iter-1)*(N+1) + epoch)
        
        print("Epoch {} : Loss {} / Auto {} / Pred {} / Line {} / State {}".format(epoch, loss, my_auto_loss, my_pred_loss, linearization_error, temp_state))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # scheduler.step(loss)
    
    with torch.no_grad():
        z_before = model.encoded(d_before)
        z_after = model.encoded(d_after)
    
        A, B, lin_err = find_AB(Z1=z_before, Z2=z_after, U1=u_list[:m], c=c, n=n, T=m, rho=rho, max_iter=400, device=device, tensorboard=tb, abscisse=(control_iter)*(N+1))
        control = torch.linalg.pinv(B) @ (
            model.encoded(torch.cat([dstar]*(secondary_horizon + 1)).to(device)).T\
                - \
            A @ model.encoded(d.before(T+1).to(device))[-1].T
        )
    
    return control, model