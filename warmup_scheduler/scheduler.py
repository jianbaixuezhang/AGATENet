from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

#学习率
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    # 类构造方法，用于初始化学习率预热调度器
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        if multiplier < 1.0:
            raise ValueError('Multiplier must be ≥ 1.0.')
        if total_epoch < 1:
            raise ValueError('Total epochs must be ≥ 1.')
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and not self.finished:
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_lr() if self.after_scheduler else [
                base_lr * self.multiplier for base_lr in self.base_lrs
            ]
        if self.multiplier == 1.0:
            return [base_lr * (self.last_epoch / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1) * self.last_epoch / self.total_epoch + 1)
                    for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        epoch = epoch or self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # Handle epoch 0 edge case
        if self.last_epoch <= self.total_epoch:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.after_scheduler.step(metrics, epoch - self.total_epoch if epoch else None)

    def step(self, epoch=None, metrics=None):
        if isinstance(self.after_scheduler, ReduceLROnPlateau):
            self.step_ReduceLROnPlateau(metrics, epoch)
        else:
            if self.finished and self.after_scheduler:
                self.after_scheduler.step(epoch - self.total_epoch if epoch is not None else None)
            else:
                super().step(epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
