import torch
import torch.nn as nn

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)

class CompressLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(CompressLoss, self).__init__()
        self.eps = eps
    def compute_compress_loss(self, W, Pi): # W:dxb, Pi:bxb
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)
        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()

        return compress_loss

    def forward(self, X, Y,num_classes):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            # if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss =  compress_loss
        return total_loss

def NcutLoss(W, Pi, gamma=100):
    _, n_classes = Pi.shape
    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    Vol_p = torch.diag(1.0 / torch.sqrt(torch.sum(D.mm(Pi), dim=0) + 1e-6))
    H = Pi.mm(Vol_p)
    I_res = H.t().mm(D).mm(H) - torch.eye(n_classes, device=Pi.device)

    spectral = torch.trace(H.t().mm(L).mm(H))
    orth_reg = torch.norm(I_res) / n_classes
    return spectral, orth_reg