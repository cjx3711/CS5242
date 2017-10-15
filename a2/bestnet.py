# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
from code_base.classifiers.cnn import *
from code_base.data_utils import get_CIFAR2_data
from code_base.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from code_base.layers import *
from code_base.solver import Solver

# Load the (preprocessed) CIFAR2 (airplane and bird) data.

data = get_CIFAR2_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)
  
model = ThreeLayerConvNet(num_classes=2, weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=2, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

print(solver.train_acc_history)
print(solver.loss_history)