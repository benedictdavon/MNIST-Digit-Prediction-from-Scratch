import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .Layer import Layer
from ..helpers.Backend import backend

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # store parameters, initialize weights/bias
        # in_channels -> because input image has 1 channel (e.g., grayscale MNIST digits).
        # out_channels -> the number of filters/kernels you want to learn in that layer.
        # weights: (out_channels, in_channels, kernel_size, kernel_size)
        # bias: (out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization
        fan_in = in_channels * kernel_size * kernel_size
        dtype = np.float32

        # Initialize on CPU, then move to GPU if needed
        weights_cpu = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / fan_in).astype(dtype)
        bias_cpu = np.zeros((out_channels, 1), dtype=dtype)
        
        # Move to appropriate backend (GPU/CPU)
        self.weights = backend.ensure_array(weights_cpu)
        self.bias = backend.ensure_array(bias_cpu)

        # grads (filled during backward)
        self.dW = backend.zeros_like(self.weights)
        self.db = backend.zeros_like(self.bias)

    # ----- helpers -----
    def _pad(self, x):
        if self.padding == 0:
            return x
        p = self.padding
        return backend.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    def forward(self, x, training=False):
        # x shape: (batch, in_channels, H, W)
        # return: (batch, out_channels, H_out, W_out)
        # H_out = (H + 2*padding - kernel_size)//stride + 1\
        self.x = backend.ensure_array(x)  # cache for backward and ensure on correct backend
        batch_size, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.H_out = H_out
        self.W_out = W_out

        # output tensor
        out = backend.zeros((batch_size, self.out_channels, H_out, W_out))

        # pad input
        xp = self._pad(self.x)  # (B, C, H_in+2p, W_in+2p)

        # Super fast vectorized im2col using pure backend operations (no loops!)
        batch_size, in_channels, H_pad, W_pad = xp.shape
        
        # Create all patch position indices at once
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
        
        # Extract all patches at once using advanced indexing
        # xp: (batch, in_channels, H_pad, W_pad)
        # We want: (batch, in_channels, H_out, W_out, k, k)
        batch_idx = backend.arange(batch_size)[:, None, None, None, None, None]  # (B, 1, 1, 1, 1, 1)
        channel_idx = backend.arange(in_channels)[None, :, None, None, None, None]  # (1, C, 1, 1, 1, 1)
        h_idx = h_all[None, None, :, :, :, :]  # (1, 1, H_out, W_out, k, k)
        w_idx = w_all[None, None, :, :, :, :]  # (1, 1, H_out, W_out, k, k)
        
        # Extract patches: (batch, in_channels, H_out, W_out, k, k)
        patches = xp[batch_idx, channel_idx, h_idx, w_idx]
        
        # Reshape to column format: (batch, H_out*W_out, in_channels*k*k)
        cols = backend.reshape(backend.transpose(patches, (0, 2, 3, 1, 4, 5)), (batch_size, H_out * W_out, -1))
        self.cols = cols  # cache for backward

        # Cache top-left indices for backward scatter
        ii = backend.repeat(backend.arange(H_out), W_out)  # (HW,)
        jj = backend.tile(backend.arange(W_out), H_out)  # (HW,)
        self.h0 = ii * self.stride  # (HW,)
        self.w0 = jj * self.stride  # (HW,)

        # reshape filter
        # each filter is flattened into a row vector
        # shape: (out_channels, in_channels*k*k)
        W_col = backend.reshape(self.weights, (self.out_channels, -1))  # (out_channels, in_channels*k*k)

        # matrix multiply
        # multiply all patches with all filters at once
        # cols: (batch, H_out*W_out, in_channels*k*k)
        # W_col.T: (in_channels*k*k, out_channels)
        # result: (batch, H_out*W_out, out_channels)
        out = backend.matmul(cols, backend.transpose(W_col)) + backend.transpose(self.bias)

        # reshape back
        # transpose to (batch, out_channels, H_out*W_out)
        # then reshape to (batch, out_channels, H_out, W_out)
        out = backend.reshape(backend.transpose(out, (0, 2, 1)), (batch_size, self.out_channels, H_out, W_out))

        return out

    def backward(self, grad_out):
        """
        grad_out: (B, O, H_out, W_out)
        returns:  grad_input of shape like self.x
        """
        grad_out = backend.ensure_array(grad_out)
        x = self.x
        batch_size, channels, H_in, W_in = x.shape

        H_out, W_out = self.H_out, self.W_out
        HW = H_out * W_out

        # Flatten grad_out to match forward col arrangement:
        # go: (B, HW, O)
        go = backend.reshape(grad_out, (batch_size, self.out_channels, HW))
        go = backend.transpose(go, (0, 2, 1))

        # ---- dW and db ----
        # dW_col: (O, Ck2) = sum_b ( go[b]^T @ cols[b] )
        # Use matrix operations to sum over batch & windows in one shot
        # go:   (B, HW, O)
        # cols: (B, HW, Ck2)
        # Equivalent to einsum("bho,bhc->oc", go, self.cols)
        go_reshaped = backend.reshape(go, (-1, self.out_channels))  # (B*HW, O)
        cols_reshaped = backend.reshape(self.cols, (-1, self.cols.shape[-1]))  # (B*HW, Ck2)
        dW_col = backend.matmul(backend.transpose(go_reshaped), cols_reshaped)  # (O, Ck2)
        self.dW[...] = backend.reshape(dW_col, self.weights.shape)  # (O, C, k, k)

        # db: sum over batch and windows for each output channel
        db_new = backend.sum(go, axis=(0, 1))                         # (O,)
        self.db[...] = backend.reshape(db_new, (self.out_channels, 1))  # (O,1)

        # ---- dX via fully vectorized col2im using tensor operations (no loops!) ----
        W_col = backend.reshape(self.weights, (self.out_channels, -1))  # (O, Ck2)
        cols_grad = backend.matmul(go, W_col)  # (B, HW, Ck2) - fast matrix multiplication
        
        # Use the efficient approach: unfold the gradient back to input space
        # This is mathematically equivalent to the transpose of the forward convolution
        
        # Reshape cols_grad to match the patch extraction format
        cols_grad = backend.reshape(cols_grad, (batch_size, H_out, W_out, channels, self.kernel_size, self.kernel_size))
        
        # Use correlation/convolution with flipped kernel to implement transpose
        # This is the standard approach used in frameworks like PyTorch/TensorFlow
        
        # For simplicity and to ensure correctness, we'll use the slice-based approach
        # but vectorize it as much as possible
        H_pad = H_in + 2 * self.padding
        W_pad = W_in + 2 * self.padding
        grad_xp = backend.zeros((batch_size, channels, H_pad, W_pad))
        
        # Convert to CPU for slicing operations that are tricky in CuPy
        if backend.use_gpu:
            cols_grad_cpu = backend.to_cpu(cols_grad)
            grad_xp_cpu = backend.to_cpu(grad_xp)
            
            # Vectorized slice assignment - this is the fastest reliable approach
            # Each iteration handles a different spatial position but all batches/channels at once
            for h_idx in range(H_out):
                for w_idx in range(W_out):
                    h_start = h_idx * self.stride
                    w_start = w_idx * self.stride
                    h_end = h_start + self.kernel_size
                    w_end = w_start + self.kernel_size
                    
                    # Vectorized over batch and channel dimensions
                    grad_xp_cpu[:, :, h_start:h_end, w_start:w_end] += np.transpose(cols_grad_cpu[:, h_idx, w_idx], (0, 1, 2, 3))
            
            grad_xp = backend.ensure_array(grad_xp_cpu)
        else:
            # Vectorized slice assignment - this is the fastest reliable approach
            # Each iteration handles a different spatial position but all batches/channels at once
            for h_idx in range(H_out):
                for w_idx in range(W_out):
                    h_start = h_idx * self.stride
                    w_start = w_idx * self.stride
                    h_end = h_start + self.kernel_size
                    w_end = w_start + self.kernel_size
                    
                    # Vectorized over batch and channel dimensions
                    grad_xp[:, :, h_start:h_end, w_start:w_end] += backend.transpose(cols_grad[:, h_idx, w_idx], (0, 1, 2, 3))

        # Remove padding
        if self.padding > 0:
            grad_in = grad_xp[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_in = grad_xp

        return grad_in

    # expose params / grads for the optimizer
    def params(self):
        return [self.weights, self.bias]

    def grads(self):
        return [self.dW, self.db]

