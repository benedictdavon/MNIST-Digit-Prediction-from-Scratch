import numpy as np
from .Layer import Layer
from ..helpers.Backend import backend

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, training=False):
        # x shape: (batch, channels, H, W)
        # return: (batch, channels, H_out, W_out)
        batch_size, channels, H_in, W_in = x.shape

        # Need to store input and max indices during forward pass
        # Store input for backward pass and ensure on correct backend
        self.x = backend.ensure_array(x)

        # compute output shape
        H_out = (H_in - self.kernel_size) // self.stride + 1
        W_out = (W_in - self.kernel_size) // self.stride + 1

        self.H_out = H_out
        self.W_out = W_out

        # output tensor
        out = backend.zeros((batch_size, channels, H_out, W_out))

        # # Loop version (clear but slow)
        # for n in range(batch_size):
        #     for c in range(channels):
        #         for i in range(H_out):
        #             for j in range(W_out):
        #                 h_start = i * self.stride
        #                 h_end   = h_start + self.kernel_size
        #                 w_start = j * self.stride
        #                 w_end   = w_start + self.kernel_size
        #
        #                 region = x[n, c, h_start:h_end, w_start:w_end]
        #                 out[n, c, i, j] = np.max(region)

        # Super fast vectorized pooling using pure backend operations (no loops!)
        batch_size, channels, H_in, W_in = self.x.shape
        
        # Create all pool position indices at once
        h_out_indices = backend.arange(H_out)  # (H_out,)
        w_out_indices = backend.arange(W_out)  # (W_out,)
        h_grid, w_grid = backend.meshgrid(h_out_indices, w_out_indices, indexing='ij')
        
        # Broadcast to get all starting positions: (H_out, W_out)
        h_starts = h_grid * self.stride  # (H_out, W_out)
        w_starts = w_grid * self.stride  # (H_out, W_out)
        
        # Create kernel offset indices
        k_indices = backend.arange(self.kernel_size)  # (k,)
        kh_grid, kw_grid = backend.meshgrid(k_indices, k_indices, indexing='ij')
        
        # Broadcast to get all absolute positions
        # h_starts: (H_out, W_out) -> (H_out, W_out, 1, 1)
        # kh_grid: (k, k) -> (1, 1, k, k)
        h_all = h_starts[:, :, None, None] + kh_grid[None, None, :, :]  # (H_out, W_out, k, k)
        w_all = w_starts[:, :, None, None] + kw_grid[None, None, :, :]  # (H_out, W_out, k, k)
        
        # Extract all pooling windows at once using advanced indexing
        # x: (batch, channels, H_in, W_in)
        # We want: (batch, channels, H_out, W_out, k, k)
        batch_idx = backend.arange(batch_size)[:, None, None, None, None, None]  # (B, 1, 1, 1, 1, 1)
        channel_idx = backend.arange(channels)[None, :, None, None, None, None]  # (1, C, 1, 1, 1, 1)
        h_idx = h_all[None, None, :, :, :, :]  # (1, 1, H_out, W_out, k, k)
        w_idx = w_all[None, None, :, :, :, :]  # (1, 1, H_out, W_out, k, k)
        
        # Extract pooling windows: (batch, channels, H_out, W_out, k, k)
        windows = self.x[batch_idx, channel_idx, h_idx, w_idx]
        
        # Reshape for pooling operations: (batch, channels, H_out*W_out, k*k)
        cols = backend.reshape(backend.transpose(windows, (0, 1, 2, 3, 4, 5)), (batch_size, channels, H_out * W_out, -1))

        # cache argmax positions inside each window (flattened index 0..k*k-1)
        self.argmax = backend.argmax(cols, axis=-1)  # (B, C, HW)

        # pooled output
        out = backend.reshape(backend.max(cols, axis=-1), (batch_size, channels, H_out, W_out))  # (B, C, H_out, W_out)
        return out

    def backward(self, grad_out):
        """
        Vectorized backward pass with minimal loops
        """
        grad_out = backend.ensure_array(grad_out)
        x = self.x
        B, C, H, W = x.shape
        k, s = self.kernel_size, self.stride
        H_out, W_out = self.H_out, self.W_out

        # Flatten grad_out and argmax
        go = backend.reshape(grad_out, (B, C, H_out, W_out))  # (B, C, H_out, W_out)
        p = backend.reshape(self.argmax, (B, C, H_out, W_out))  # (B, C, H_out, W_out)

        # Convert flattened argmax -> (u, v) offsets within k x k window
        u = p // k  # (B, C, H_out, W_out)
        v = p % k   # (B, C, H_out, W_out)

        # Initialize gradient buffer
        grad_x = backend.zeros_like(x)

        # For GPU operations, handle the loops on CPU side and use vectorized operations
        if backend.use_gpu:
            go_cpu = backend.to_cpu(go)
            u_cpu = backend.to_cpu(u)
            v_cpu = backend.to_cpu(v)
            grad_x_cpu = backend.to_cpu(grad_x)
            
            # More vectorized approach: handle all windows for each position
            # This reduces the number of loop iterations significantly
            for h_idx in range(H_out):
                for w_idx in range(W_out):
                    h_start = h_idx * s
                    w_start = w_idx * s
                    
                    # Get the argmax positions and gradients for this window (all batches/channels)
                    u_win = u_cpu[:, :, h_idx, w_idx]  # (B, C)
                    v_win = v_cpu[:, :, h_idx, w_idx]  # (B, C)
                    go_win = go_cpu[:, :, h_idx, w_idx]  # (B, C)
                    
                    # Compute absolute positions (vectorized over batch/channel)
                    h_abs = h_start + u_win  # (B, C)
                    w_abs = w_start + v_win  # (B, C)
                    
                    # Use advanced indexing to assign gradients (vectorized over B,C)
                    batch_indices = np.arange(B)[:, None]  # (B, 1)
                    channel_indices = np.arange(C)[None, :]  # (1, C)
                    
                    grad_x_cpu[batch_indices, channel_indices, h_abs, w_abs] += go_win
            
            grad_x = backend.ensure_array(grad_x_cpu)
        else:
            # More vectorized approach: handle all windows for each position
            # This reduces the number of loop iterations significantly
            for h_idx in range(H_out):
                for w_idx in range(W_out):
                    h_start = h_idx * s
                    w_start = w_idx * s
                    
                    # Get the argmax positions and gradients for this window (all batches/channels)
                    u_win = u[:, :, h_idx, w_idx]  # (B, C)
                    v_win = v[:, :, h_idx, w_idx]  # (B, C)
                    go_win = go[:, :, h_idx, w_idx]  # (B, C)
                    
                    # Compute absolute positions (vectorized over batch/channel)
                    h_abs = h_start + u_win  # (B, C)
                    w_abs = w_start + v_win  # (B, C)
                    
                    # Use advanced indexing to assign gradients (vectorized over B,C)
                    batch_indices = backend.arange(B)[:, None]  # (B, 1)
                    channel_indices = backend.arange(C)[None, :]  # (1, C)
                    
                    grad_x[batch_indices, channel_indices, h_abs, w_abs] += go_win

        return grad_x

    def params(self):
        return []

    def grads(self):
        return []
