import numpy as np
import torch
import torch.nn as nn
from points import trainingData
import time
import scipy.io
from scipy.io import savemat
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class PINN_Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.hiddenLayers = [nn.Linear(layers[0], layers[2])] + [nn.Linear(layers[2], layers[2]) for i in
                                                                 range(layers[1] - 1)] + [
                                nn.Linear(layers[2], layers[3])]
        self.linears = nn.ModuleList(self.hiddenLayers)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        for i in range(len(self.linears)):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
        self.iter = 0
        self.layers = layers

    def forward(self, x, y):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).to(device)
        else:
            x = x.clone()
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y).to(device)
        else:
            y = y.clone()
        a = torch.cat((x, y), 1)
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    # Dirichlet边界
    def Dirichlet(self, bc_D_x, bc_D_y, bc_D_p):
        predict = self.forward(bc_D_x, bc_D_y)
        p = predict[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_Dirichlet = mse(p, bc_D_p)
        return mse_Dirichlet

    # Neumann边界
    def Neumann(self, bc_N_x, bc_N_y):
        predict = self.forward(bc_N_x, bc_N_y)
        v = predict[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((bc_N_x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_Neumann = mse(v, batch_t_zeros)
        return mse_Neumann

    # 类内方法：求方程点的loss
    def equation_mse(self, x, y, k, m):
        predict_out = self.forward(x, y)
        # 获得预测的输出psi,p
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        u_l2 = torch.sqrt(u ** 2 + v ** 2)
        # 计算偏微分方程的残差
        f_equation_x = k * u + k*u_l2 * u + p_x
        f_equation_y = k * v + k*u_l2 * v + p_y
        f_mass = u_x + v_y
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        loss_equation = m(f_equation_x, batch_t_zeros) + m(f_equation_y, batch_t_zeros)
        loss_fass = m(f_mass, batch_t_zeros)
        mse_equation = loss_equation + loss_fass
        return loss_equation, loss_fass, mse_equation

    def test(self, x, y, u, v, p):
        predict_out = self.forward(x, y)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        u_predict = torch.cat((u_predict, v_predict), dim=1)
        u_pred = u_predict.cpu().detach().numpy()
        p_pred = p_predict.cpu().detach().numpy()
        u_test = torch.cat((u, v), dim=1)
        u_test = u_test.cpu().detach().numpy()
        p_test = p.cpu().detach().numpy()
        error_u = np.linalg.norm((u_test - u_pred), 2) / np.linalg.norm(u_test, 2)
        error_p = np.linalg.norm((p_test - p_pred), 2) / np.linalg.norm(p_test, 2)
        return error_u, error_p, u_pred, p_pred


data_x = scipy.io.loadmat('data/X.mat')
data = scipy.io.loadmat('data/v1_100_kappa1.mat')
X = data_x['X']
Y = data_x['Y']
U = data['v1_100_kappa1']
V = data['v2_100_kappa1']
P = data['p_100_kappa1']
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32)
u = torch.tensor(U, dtype=torch.float32)
v = torch.tensor(V, dtype=torch.float32)
p = torch.tensor(P, dtype=torch.float32)
X_total = torch.cat([x, y, u, v, p], 1)
X_total_arr = X_total.data.numpy()
X_random = torch.tensor(X_total_arr)
del x, y, u, v, p
x_star = X_random[:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
y_star = X_random[:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
u_star = X_random[:, 2].reshape(-1, 1).clone().requires_grad_(True).to(device)
v_star = X_random[:, 3].reshape(-1, 1).clone().requires_grad_(True).to(device)
p_star = X_random[:, 4].reshape(-1, 1).clone().requires_grad_(True).to(device)

num_bc = 100
num_f = 10000
bc_D_xy, bc_D_p, bc_N_xy, all_xy_train = trainingData(num_bc, num_f)


layer_mat = [2, 8, 80, 3]
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
adam_optimizer = torch.optim.Adam(pinn_net.parameters(), lr=0.001)
lbfgs_optimizer = torch.optim.LBFGS(pinn_net.parameters(), lr=0.1)

epochs = 20000
lbfgs_epochs = 5000

mse = torch.nn.MSELoss()
mean = torch.nn.MSELoss(reduction='none')

bc_D_x = bc_D_xy[:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
bc_D_y = bc_D_xy[:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
bc_D_p = bc_D_p.reshape(-1, 1).clone().requires_grad_(True).to(device)
bc_N_x = bc_N_xy[:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
bc_N_y = bc_N_xy[:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
x_train = all_xy_train[:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
y_train = all_xy_train[:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)


def k_exact(x,y):
    condition = (x > 0.4) & (x < 0.6) & (y > 0.4) & (y < 0.6)
    result = torch.where(condition, torch.tensor(0.01), torch.tensor(1))
    return result


k = k_exact(x_train, y_train)

start_time = time.time()
for epoch in range(epochs):
    mse_Dirichlet = pinn_net.Dirichlet(bc_D_x, bc_D_y, bc_D_p)
    mse_Neumann = pinn_net.Neumann(bc_N_x, bc_N_y)
    loss_equation, loss_fass, mse_equation = pinn_net.equation_mse(x_train, y_train, k, mse)
    loss = mse_Dirichlet + mse_Neumann + mse_equation
    print("Epoch:", (epoch + 1), " Training Loss:", loss.data)
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()


def closure():
    mse_Dirichlet = pinn_net.Dirichlet(bc_D_x, bc_D_y, bc_D_p)
    mse_Neumann = pinn_net.Neumann(bc_N_x, bc_N_y)
    loss_equation, loss_fass, mse_equation = pinn_net.equation_mse(x_train, y_train, k, mse)
    loss = mse_Dirichlet + mse_Neumann + mse_equation
    print(loss, loss_equation, loss_fass)
    lbfgs_optimizer.zero_grad()
    loss.backward()
    return loss


for epoch in range(lbfgs_epochs):
    lbfgs_optimizer.step(closure)


error_u, error_p, u_pred, p_pred = pinn_net.test(x_star, y_star, u_star, v_star, p_star)
print('u_test Error: %.5f' % (error_u), 'p_test Error: %.5f' % (error_p))

