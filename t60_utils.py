import torch
from asteroid.losses import PITLossWrapper


class newPITLossWrapper(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None, reduction=True):
        super(newPITLossWrapper, self).__init__(loss_func, pit_from, perm_reduce)
        self.reduction = reduction

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        if self.reduction:
            mean_loss = torch.mean(min_loss)
        else:
            mean_loss = min_loss
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        return mean_loss, reordered


def makelambda(name):
    def getlambda2(val):
        if not isinstance(val, torch.Tensor):
            val = torch.stack(val)
        return torch.e ** (val / 10)

    def getlambda3(val):
        if not isinstance(val, torch.Tensor):
            val = torch.stack(val)
        val = torch.e ** (1 - val / 10)
        val = val / (1 + val)
        return val

    if 'lambdaloss2' in name or 'lambda2' in name or 'lambda1' in name or 'lambdaloss1' in name:
        return getlambda2
    elif 'lambdaloss3' in name or 'lambda3' in name:
        return getlambda3