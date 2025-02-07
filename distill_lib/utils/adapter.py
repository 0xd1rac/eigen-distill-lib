import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        """
        A simple adapter to map teacher feature maps into the student feature space.
        
        Args:
            in_channels (int): Number of channels in the teacher feature maps.
            out_channels (int): Desired number of channels (student feature maps).
            kernel_size (int): Kernel size for the convolution; 1 is common for linear mapping.
        """
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        
    def forward(self, x):
        return self.adapter(x)
