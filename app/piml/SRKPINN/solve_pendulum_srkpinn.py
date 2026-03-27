###############################################################################
## SRKPINN example: nonlinear pendulum
###############################################################################

import os
import sys

sys.path.append(".")

import numpy as np
import torch.nn as nn

from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.utils.common import Timer, set_seed
from ai4plasma.utils.device import check_gpu
from SRKPINN import HamiltonianSRKPINN, PendulumSystem, SRKPINNVisCallback


set_seed(2026)

if check_gpu(print_required=True):
    DEVICE.set_device(0)
else:
    DEVICE.set_device(-1)
print(DEVICE)

timer = Timer()


# ============================================================================
# Problem setup
# ============================================================================

system = PendulumSystem(
    omega0=1.0,
    q_range=(-np.pi, np.pi),
    p_range=(-2.0, 2.0),
)

dt = 0.1
stages = 2
method = "gauss-legendre"


# ============================================================================
# Model configuration
# ============================================================================

num_epochs = 6000
learning_rate = 1e-3
train_data_size = 512

layers = [system.state_dim, 128, 128, 128, system.state_dim * stages]
backbone_net = FNN(layers=layers, act_fun=nn.Tanh())

model = HamiltonianSRKPINN(
    system=system,
    dt=dt,
    stages=stages,
    method=method,
    backbone_net=backbone_net,
    train_data_size=train_data_size,
    sample_mode="uniform",
    loss_weights={
        "StageDynamics": 1.0,
        "InitialOrData": 2.0,
    },
)


# ============================================================================
# Monitoring
# ============================================================================

results_dir = "app/piml/SRKPINN/results/pendulum/"
log_dir = "app/piml/SRKPINN/runs/pendulum/"
checkpoint_dir = "app/piml/SRKPINN/models/pendulum/"
os.makedirs(results_dir, exist_ok=True)

viz_callback = SRKPINNVisCallback(
    model=model,
    initial_state=np.array([1.7, 0.0], dtype=REAL()),
    num_rollout_steps=200,
    log_freq=100,
    save_history=True,
    history_freq=100,
)
model.register_visualization_callback(viz_callback)


# ============================================================================
# Training
# ============================================================================

model.create_optimizer("Adam", lr=learning_rate)
model.create_lr_scheduler("MultiStepLR", milestones=[2000, 4000], gamma=0.5)

model.train(
    num_epochs=num_epochs,
    print_loss=True,
    print_loss_freq=100,
    tensorboard_logdir=log_dir,
    save_final_model=True,
    checkpoint_dir=checkpoint_dir,
    checkpoint_freq=1000,
)


# ============================================================================
# Export results
# ============================================================================

viz_callback.save_final_results(save_dir=results_dir, epoch=num_epochs)

print("\n" + "=" * 70)
print("SRKPINN TRAINING SUMMARY")
print("=" * 70)
print(f"Results saved to: {results_dir}")
print(f"TensorBoard logs: {log_dir}")
print(f"Checkpoints: {checkpoint_dir}")
print("Total training time:", end=" ")
timer.current()
print("=" * 70)
