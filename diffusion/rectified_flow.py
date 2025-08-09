import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedFlow(nn.Module):
    # ODE f(t+dt) = f(t) + dt*f'(t)
    def euler(self, x_t, v, dt):
        x_t = x_t + v * dt
        return x_t
    def create_flow(self, x_1, t, x_0=None):
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        
        t = t[:, None, None, None]
        x_t = (1-t)*x_0+t*x_1

        return x_t, x_0
    
    def mse_loss(self, v, x_1, x_0):
        loss = F.mse_loss(x_1 - x_0, v) # 优化目标是x1-x0
        return loss

if __name__ == "__main__":
    
    rf = RectifiedFlow()

    x_t = rf.create_flow(torch.ones(2, 3, 4, 4), 0.999)

    x_1, y = data  # x_1原始图像，y是标签，用于CFG
    v_pred = model(x=x_t, t=t, y=y)

    loss = rf.mse_loss(v_pred, x_1, x_0)

