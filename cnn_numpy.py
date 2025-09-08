import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # store parameters, initialize weights/bias
        # weights: (out_channels, in_channels, kernel_size, kernel_size)
        # bias: (out_channels, 1)
        pass

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        # return: (batch, out_channels, H_out, W_out)
        # H_out = (H + 2*padding - kernel_size)//stride + 1
        pass

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)
    
class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        pass
    
    def forward(self, x):
        # x shape: (batch, channels, H, W)
        # return: (batch, channels, H_out, W_out)
        pass

class Flatten:
    def __init__(self):
        pass

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        # return: (batch, channels*H*W)
        pass

class FullyConnectedLayer:
    def __init__(self, in_features, out_features):
        # initialize weights/bias
        # weights: (out_features, in_features)
        # bias: (out_features, 1)

        pass

    def forward(self, x):
        # x shape: (batch, in_features)
        # return: (batch, out_features)
        pass

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        # x shape: (batch, num_classes)
        # return: (batch, num_classes)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
