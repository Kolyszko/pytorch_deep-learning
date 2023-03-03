"""
Contains PyTorch model code to instantiate a simple CNN model.
"""
import torch

from torch import nn

class MyCNN(nn.Module):
  """Creates a simple CNN architecture.

  Args:
  input_shape: An integer indicating number of input channels.
  hidden_units: An integer indicating number of hidden units between layers.
  output_shape: An integer indicating number of output units.
  """

  def __init__(self, input_shape: int=1, hidden_units: int=1, output_shape: int=1):
    super().__init__()
    self.CNN_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, 
                  out_channels=hidden_units, 
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),

        nn.Conv2d(hidden_units,
                  hidden_units*2,
                  3,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
    
    self.CNN_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2, 
                  out_channels=hidden_units*4, 
                  kernel_size=3,
                  padding=1),
        nn.ReLU(),

        nn.Conv2d(hidden_units*4,
                  hidden_units*8,
                  3,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
    
    self.Dense_block = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(),
        nn.Linear(hidden_units*8*56*56,
                  output_shape)
    )

  def forward(self, x):
    x = self.CNN_block_1(x)
    x = self.CNN_block_2(x)
    x = self.Dense_block(x)
    return x
