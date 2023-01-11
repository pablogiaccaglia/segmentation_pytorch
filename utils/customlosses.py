# PyTorch
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha = 0.8, beta = 0.2, gamma = 2, smooth = 1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky

class CalSSLoss(nn.Module):
    def __init__(self, drloc_mode = 'l1'):
        super(CalSSLoss, self).__init__()

        self.drloc_mode = drloc_mode
        if drloc_mode == "l1":  # l1 regression constraint
            self.reld_criterion = self.relative_constraint_l1
        elif drloc_mode == "ce":  # cross entropy constraint
            self.reld_criterion = self.relative_constraint_ce
        elif drloc_mode == "cbr":  # cycle-back regression constaint: https://arxiv.org/pdf/1904.07846.pdf
            self.reld_criterion = self.relative_constraint_cbr
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")
    def relative_constraint_l1(self, deltaxy, predxy):
        return F.l1_loss(deltaxy, predxy)

    def relative_constraint_ce(self, deltaxy, predxy):
        # predx, predy = torch.chunk(predxy, chunks=2, dim=1)
        predx, predy = predxy[:, :, 0], predxy[:, :, 1]
        targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()
        return F.cross_entropy(predx, targetx) + F.cross_entropy(predy, targety)

    def variance_aware_regression(self, pred, beta, target, labels, lambda_var = 0.001):
        # Variance aware regression.
        pred_titled = pred.unsqueeze(0).t().repeat(1, labels.size(1))
        EPSILON = 1e-8
        pred_var = torch.sum((labels - pred_titled) ** 2 * beta, dim = 1) + EPSILON
        pred_log_var = torch.log(pred_var)
        squared_error = (pred - target) ** 2
        return torch.mean(torch.exp(-pred_log_var) * squared_error + lambda_var * pred_log_var)

    # based on the codes: https://github.com/google-research/google-research/blob/master/tcc/tcc/losses.py
    def relative_constraint_cbr(self, deltaxy, predxy, loss_type = "regression_mse_var"):
        predx, predy = predxy[:, :, 0], predxy[:, :, 1]
        num_classes = predx.size(1)
        targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()  # [N, ], [N, ]
        betax, betay = F.softmax(predx, dim = 1), F.softmax(predy, dim = 1)  # [N, C], [N, C]
        labels = torch.arange(num_classes).unsqueeze(0).to(predxy.device)  # [1, C]
        true_idx = targetx  # torch.sum(targetx*labels, dim=1)      # [N, ]
        true_idy = targety  # torch.sum(targety*labels, dim=1)      # [N, ]

        pred_idx = torch.sum(betax * labels, dim = 1)  # [N, ]
        pred_idy = torch.sum(betay * labels, dim = 1)  # [N, ]

        if loss_type in ["regression_mse", "regression_mse_var"]:
            if "var" in loss_type:
                # Variance aware regression.
                lossx = self.variance_aware_regression(pred_idx, betax, true_idx, labels)
                lossy = self.variance_aware_regression(pred_idy, betay, true_idy, labels)
            else:
                lossx = torch.mean((pred_idx - true_idx) ** 2)
                lossy = torch.mean((pred_idy - true_idy) ** 2)
            loss = lossx + lossy
            return loss
        else:
            raise NotImplementedError("We only support regression_mse and regression_mse_var now.")

    def forward(self, outs, lambda_drloc = 0.0):
        loss, all_losses = 0.0, Munch()
        loss_drloc = 0.0
        for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
            loss_drloc += self.reld_criterion(deltaxy, drloc) * lambda_drloc
        all_losses.drloc = loss_drloc.item()
        loss += loss_drloc

        return loss, all_losses
