import torch
import torch.nn as nn
import numpy as np

class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SGRR(nn.Module):
    def __init__(self, in_dim, hid_dim, feat_dim, cluster_dim):
        super(SGRR, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.feat_dim = feat_dim
        self.cluster_dim = cluster_dim
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.hid_dim),
                                         nn.BatchNorm1d(self.hid_dim),
                                         nn.ReLU(),
                                        )
        self.feature_head = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hid_dim, self.feat_dim))
        self.cluster_head = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hid_dim, self.cluster_dim))

        self.thres = AdaptiveSoftThreshold(1)
        self.shrink = 1.0 / self.cluster_dim

    def get_aff(self, x):
        pre_feat = self.projector(x)
        feat = self.feature_head(pre_feat)
        aff = self.thres(feat.mm(feat.T))
        return self.shrink * aff

    def forward(self, x):
        pre_feat = self.projector(x)
        feat = self.feature_head(pre_feat)
        logits = self.cluster_head(pre_feat)
        return feat, logits


class Gumble_Softmax(nn.Module):
    def __init__(self,tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through
    
    def forward(self,logits):
        logps = torch.log_softmax(logits,dim=1)
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits/self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits*1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out


class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, c):
        C = -c
        x_points = C.shape[-2]
        y_points = C.shape[-1]
        batch_size = C.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-12

        # Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))

        return pi

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1