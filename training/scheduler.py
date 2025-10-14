"""
Learning Rate Schedulers

Provides various learning rate scheduling strategies for training.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on config

    Args:
        optimizer: PyTorch optimizer
        config: training config dict with scheduler parameters

    Returns:
        scheduler: PyTorch learning rate scheduler
    """
    scheduler_type = config.get('scheduler_type', 'cosine')
    num_epochs = config.get('num_epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 5)

    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=config.get('min_lr', 1e-6)
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=config.get('min_lr', 1e-6)
            )

    elif scheduler_type == 'linear':
        # Linear warmup + decay
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_epochs
        )

    elif scheduler_type == 'constant':
        # Constant learning rate (no scheduling)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class WarmupCosineScheduler:
    """
    Custom warmup + cosine decay scheduler

    Provides more control over warmup and decay phases.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, max_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Test config
    config = {
        'scheduler_type': 'cosine',
        'num_epochs': 100,
        'warmup_epochs': 10,
        'min_lr': 1e-6
    }

    scheduler = get_scheduler(optimizer, config)

    # Simulate training
    lrs = []
    for epoch in range(config['num_epochs']):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    print("Learning rate schedule saved to lr_schedule.png")
