import torch
from sys import stdout
import numpy as np
import time

class MechanicalRegression:
    def __init__(self, pb_type, L, activ, p, D2, nu=0.01, gamma2=0.01):
        self.pb_type = pb_type
        self.L = L
        self.activ = activ
        self.alpha1_init = torch.zeros((p+1, p, L), dtype=torch.float64)
        self.D2 = D2
        self.gamma2 = gamma2
        self.nu = nu
        self.dt = 1/L if L > 0 else 0
     
    def psi(self, x, D = None):
        if D is None:
            z = x
        else:
            np.random.seed(11)
            p = x.shape[1]
            W = torch.from_numpy(1.5 * np.divide(np.random.randn(p, D), np.sqrt(p)))
            bias = torch.from_numpy(0.1 * np.random.randn(D))
            z = torch.matmul(x, W) + bias
            
        if self.activ == "ReLU":
            z = torch.clamp(z, min=0)
        if self.activ == "sigmoid":
            z = 1 / (1 + torch.exp(-z))
        if self.activ == "tanh":
            z = torch.tanh(z)
            
        if D is None:
            z = torch.cat([z, torch.ones((x.shape[0], 1))], dim=1)
        return z

    def psi1(self, x):
        return self.psi(x)

    def psi2(self, x):
        return self.psi(x, self.D2)

    def phi(self, x, alpha1):
        """propagate inputs across L layers defined by the 3D tensor alpha1"""
        for i in range(self.L):
            x_psi = self.psi1(x)
            x = x + torch.matmul(x_psi, alpha1[:, :, i])
        return x

    def get_alpha2(self, z, y):
        """coefficient alpha2 is obtained by solving a ridge regression"""
        zt = torch.transpose(z, 0, 1)
        A = torch.matmul(zt, z) + len(y)*self.gamma2*torch.eye(self.D2)
        b = torch.matmul(zt, y)
        alpha2 = torch.linalg.solve(A, b)
        return alpha2  
    
    def get_loss(self, x, y, alpha1=None):
        if alpha1 is None:
            alpha1 = self.alpha1_init
        loss1 = self.nu * 0.5 * self.L * torch.sum(alpha1 * alpha1)
        x_phi = self.phi(x, alpha1)
        x_psi2 = self.psi2(x_phi)
        alpha2 = self.get_alpha2(x_psi2, y)
        loss2 = torch.mean((torch.matmul(x_psi2, alpha2) - y) ** 2)
        loss2 += self.gamma2 * torch.sum(alpha2**2)
        return loss1 + loss2  
    
    def get_pred(self, x):
        """Get prediction after optimizing model"""
        x_phi = self.phi(x, self.alpha1_opt)
        x_psi2 = self.psi2(x_phi)
        pred = torch.matmul(x_psi2, self.alpha2_opt)
        if self.pb_type == 'regression':
            return pred
        else:
            return torch.argmax(pred, axis=1)
        
    def get_mse(self, x, y):
        """Get RMSE after optimizing model (only if pb_type is 'regression')"""
        return torch.sqrt(torch.mean((self.get_pred(x) - y)**2))
    
    def get_accuracy_error(self, x, y):
        """Get 1-accuracy after optimizing model (only if pb_type is 'classification')"""
        true_labels = torch.argmax(y, axis=1)
        pred_labels = self.get_pred(x)
        correct=(pred_labels == true_labels)
        return 1 - np.mean(correct.numpy())
        
    def get_error(self, x, y):
        if self.pb_type == 'regression':
            return self.get_mse(x, y)
        else:
            return self.get_accuracy_error(x, y)
    
    def fit(self, x, y, nsteps=1000, lr=1e-3):
        if self.L > 0:
            return self.fit_alpha1(x, y, nsteps, lr)
        else:
            return self.fit_alpha2(x, y)
    
    def fit_alpha1(self, x, y, nsteps=1000, lr=1e-3):
        """Optimize the propagation of inputs across layers"""
        alpha1_opt = self.alpha1_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([alpha1_opt], lr=lr)
        self.loss_history = []
        for i in range(nsteps):
            optimizer.zero_grad()
            stdout.write("\r[%s]" % (i + 1))
            loss = self.get_loss(x, y, alpha1_opt)
            self.loss_history.append(loss.detach().numpy())
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
        self.alpha1_opt = alpha1_opt.detach()
        self.alpha2_opt = self.get_alpha2(self.psi2(self.phi(x, alpha1_opt.detach())), y)
        return self.alpha1_opt
    
    def fit_alpha2(self, x, y):
        """Get the ridge regression parameter"""
        self.alpha1_opt = self.alpha1_init
        self.alpha2_opt = self.get_alpha2(self.psi2(x), y)
        return self.alpha1_opt
        













