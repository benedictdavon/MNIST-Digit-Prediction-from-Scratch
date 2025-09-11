# models/helpers/Backend.py
import os
import numpy as np

VERBOSE_STARTUP = False  # set False to reduce console noise

try:
    import cupy as cp
    if VERBOSE_STARTUP:
        print("CuPy:", cp.__version__)
        print("GPU count:", cp.cuda.runtime.getDeviceCount())
        print("Driver ver:", cp.cuda.runtime.driverGetVersion())
        print("Runtime ver:", cp.cuda.runtime.runtimeGetVersion())
        print("=================================================")
        try:
            cp.show_config()
        except Exception:
            pass
        print("=================================================")
    # Quick runtime check
    try:
        _ = (cp.array([1, 2, 3]) + 1).sum()
        CUPY_AVAILABLE = True
        if VERBOSE_STARTUP:
            print("CuPy is available - GPU acceleration enabled")
    except Exception as e:
        print(f"CuPy installed but CUDA runtime error: {e}")
        print("Falling back to CPU (NumPy)")
        cp = None
        CUPY_AVAILABLE = False
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    if VERBOSE_STARTUP:
        print("CuPy not available - using NumPy (CPU)")


class Backend:
    """Backend abstraction for NumPy/CuPy compatibility."""
    def __init__(self, use_gpu=True, default_float=np.float32):
        self.use_gpu = bool(use_gpu and CUPY_AVAILABLE)
        self.default_float = default_float
        if self.use_gpu:
            try:
                self.xp = cp
                print("Using GPU backend (CuPy)")
                if VERBOSE_STARTUP:
                    print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s) available")
            except Exception as e:
                print(f"Failed to initialize GPU backend: {e}")
                print("Falling back to CPU backend")
                self.use_gpu = False
                self.xp = np
                print("Using CPU backend (NumPy)")
        else:
            self.xp = np
            print("Using CPU backend (NumPy)")

    # -------- device transfer --------
    def to_gpu(self, x):
        """Move array to GPU if using CuPy."""
        if self.use_gpu:
            if isinstance(x, np.ndarray):
                return cp.asarray(x)
            # if it's already cupy, or not array-like, just return
            return x
        return x

    def to_cpu(self, x):
        """Move array to CPU (NumPy)."""
        if self.use_gpu and x is not None:
            # cupy.ndarray has .get() and cp.asnumpy; prefer asnumpy
            try:
                return cp.asnumpy(x)
            except Exception:
                if hasattr(x, "get"):
                    try:
                        return x.get()
                    except Exception:
                        return x
        return x

    def ensure_array(self, x, dtype=None, copy=False):
        """
        Ensure 'x' is an array of the current backend.
        Accepts list/tuple/np/cp arrays; returns xp.ndarray.
        """
        target_xp = cp if self.use_gpu else np
        if isinstance(x, target_xp.ndarray):
            # already correct backend
            if dtype is not None and x.dtype != dtype:
                return x.astype(dtype, copy=copy)
            return x
        # If it's the other backend array:
        if self.use_gpu and isinstance(x, np.ndarray):
            arr = cp.asarray(x)
            return arr.astype(dtype, copy=copy) if dtype is not None else arr
        if (not self.use_gpu) and (cp is not None) and isinstance(x, cp.ndarray):
            arr = cp.asnumpy(x)
            return arr.astype(dtype, copy=copy) if dtype is not None else arr
        # If it's list/tuple/other array-like:
        arr = target_xp.asarray(x)
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr

    def asarray(self, x, dtype=None):
        """Explicit asarray using current backend."""
        return self.ensure_array(x, dtype=dtype)

    def astype_default(self, x):
        """Cast to default float dtype if needed."""
        if hasattr(x, "dtype") and x.dtype == self.default_float:
            return x
        return self.ensure_array(x, dtype=self.default_float)

    # -------- fall back to CPU on GPU failure --------
    def _fallback_to_cpu(self):
        """Switch to CPU backend when GPU operations fail."""
        self.use_gpu = False
        self.xp = np
        print("Switched to CPU backend (NumPy)")

    # -------- array creation --------
    def array(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.default_float)
        try:
            return self.xp.array(*args, **kwargs)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.array(*args, **kwargs)
            raise

    def zeros(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.default_float)
        try:
            return self.xp.zeros(*args, **kwargs)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.zeros(*args, **kwargs)
            raise

    def ones(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.default_float)
        try:
            return self.xp.ones(*args, **kwargs)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.ones(*args, **kwargs)
            raise

    def zeros_like(self, x):
        try:
            return self.xp.zeros_like(x)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.zeros_like(self.to_cpu(x))
            raise

    def ones_like(self, x):
        try:
            return self.xp.ones_like(x)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.ones_like(self.to_cpu(x))
            raise

    def empty(self, *args, **kwargs):
        kwargs.setdefault("dtype", self.default_float)
        try:
            return self.xp.empty(*args, **kwargs)
        except Exception as e:
            if self.use_gpu:
                print(f"GPU operation failed, falling back to CPU: {e}")
                self._fallback_to_cpu()
                return np.empty(*args, **kwargs)
            raise

    # -------- math / linalg (thin wrappers) --------
    def sqrt(self, x):      return self.xp.sqrt(x)
    def maximum(self, a, b):return self.xp.maximum(a, b)
    def minimum(self, a, b):return self.xp.minimum(a, b)
    def sum(self, x, axis=None, keepdims=False):  return self.xp.sum(x, axis=axis, keepdims=keepdims)
    def mean(self, x, axis=None, keepdims=False): return self.xp.mean(x, axis=axis, keepdims=keepdims)
    def max(self, x, axis=None, keepdims=False):  return self.xp.max(x, axis=axis, keepdims=keepdims)
    def min(self, x, axis=None, keepdims=False):  return self.xp.min(x, axis=axis, keepdims=keepdims)
    def argmax(self, x, axis=None):               return self.xp.argmax(x, axis=axis)
    def argmin(self, x, axis=None):               return self.xp.argmin(x, axis=axis)
    def exp(self, x):                              return self.xp.exp(x)
    def log(self, x):                              return self.xp.log(x)
    def abs(self, x):                              return self.xp.abs(x)
    def transpose(self, x, axes=None):            return self.xp.transpose(x, axes)
    def reshape(self, x, shape):                   return self.xp.reshape(x, shape)
    def flatten(self, x):                          return self.xp.ravel(x)
    def concatenate(self, arrays, axis=0):         return self.xp.concatenate(arrays, axis=axis)
    def stack(self, arrays, axis=0):               return self.xp.stack(arrays, axis=axis)
    def dot(self, a, b):                           return self.xp.dot(a, b)
    def matmul(self, a, b):                        return self.xp.matmul(a, b)
    def arange(self, *args, **kwargs):             return self.xp.arange(*args, **kwargs)
    def meshgrid(self, *args, **kwargs):           return self.xp.meshgrid(*args, **kwargs)
    def tile(self, x, reps):                       return self.xp.tile(x, reps)
    def repeat(self, x, repeats, axis=None):       return self.xp.repeat(x, repeats, axis=axis)

    # -------- sliding window (conv helper) --------
    def sliding_window_view(self, x, window_shape, axis=None, writeable=False):
        """
        Device-aware sliding_window_view.
        Tries xp.lib.stride_tricks first. If unavailable on GPU, falls back to CPU (copy).
        """
        try:
            return self.xp.lib.stride_tricks.sliding_window_view(
                x, window_shape, axis=axis, writeable=writeable
            )
        except Exception as e:
            # As a safety net only. This incurs a device transfer + copy.
            if self.use_gpu:
                x_cpu = self.to_cpu(x)
                v = np.lib.stride_tricks.sliding_window_view(
                    x_cpu, window_shape, axis=axis, writeable=writeable
                )
                return cp.asarray(v)
            # pure CPU path
            return np.lib.stride_tricks.sliding_window_view(
                x, window_shape, axis=axis, writeable=writeable
            )

    # -------- randomness / padding --------
    @property
    def random(self):
        return self.xp.random

    def seed(self, seed=42):
        """Seed RNG for reproducibility."""
        if self.use_gpu:
            try:
                cp.random.seed(seed)
            except Exception:
                pass
        np.random.seed(seed)  # keep NumPy seeded too (for indices, etc.)

    def pad(self, array, pad_width, mode="constant", **kwargs):
        return self.xp.pad(array, pad_width, mode=mode, **kwargs)

    # -------- GPU memory / sync --------
    def clear_cache(self):
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def memory_info(self):
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            return {
                "used_bytes": mempool.used_bytes(),
                "total_bytes": mempool.total_bytes(),
                "free_bytes": mempool.total_bytes() - mempool.used_bytes(),
            }
        return None

    def synchronize(self):
        """Block until all queued GPU kernels complete (for timing)."""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

    # -------- delegate unknown attrs to xp --------
    def __getattr__(self, name):
        return getattr(self.xp, name)


# Global backend instance - can be overridden
backend = Backend(use_gpu=True)
