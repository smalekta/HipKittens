import torch

def init_randint(shape, low, high, dtype, device):
    return torch.randint(low, high, shape, dtype=dtype, device=device)

def init_randn(shape, dtype, device, scale=1):
    return scale * torch.randn(shape, dtype=dtype, device=device)

def init_empty(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)

def init_zero(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)

def init_trig(shape, dtype, device):
    M, N = shape[0], shape[1] if len(shape) > 1 else 1
    base = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(shape).sin()
    return base.to(dtype)

def print_title(title, len=30):
    print("-"*len)
    print(title)
    print("-"*len)

def bench_gemm(gemm_params, gemm_func, flush=None, transpose_B=False, num_warmup=500, num_iter=500, l2_cache_size_mb=256, mall_size_mb=512):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    m, n, k = gemm_params["shape"]
    flop = 2*m*n*k
    dtype = gemm_params["dtype"]
    device = gemm_params["device"]

    A_shape = (m, k)
    B_shape = (n, k) if transpose_B else (k, n)
    C_shape = (m, n)

    # Calculate bytes per element for the dtype
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    
    # Calculate total bytes for one set of A, B, C
    A_bytes = m * k * dtype_size
    B_bytes = B_shape[0] * B_shape[1] * dtype_size
    C_bytes = m * n * dtype_size
    set_bytes = A_bytes + B_bytes + C_bytes
    
    # Calculate number of buffer sets needed to exceed L2 cache
    l2_cache_bytes = l2_cache_size_mb * 1024 * 1024
    num_buffers_l2 = max(2, (l2_cache_bytes // set_bytes) + 2)
    
    mall_bytes = mall_size_mb * 1024 * 1024
    num_buffers_mall = max(2, (mall_bytes // set_bytes) + 2)
    
    num_buffers = max(num_buffers_l2, num_buffers_mall)
    
    # Pre-allocate rotating buffers
    A_buffers = [init_trig(A_shape, dtype, device) for _ in range(num_buffers)]
    B_buffers = [init_trig(B_shape, dtype, device) for _ in range(num_buffers)]
    C_buffers = [init_empty(C_shape, dtype, device) for _ in range(num_buffers)]

    # Warmup with first buffer set
    for _ in range(num_warmup):
        gemm_func(A_buffers[0], B_buffers[0], C_buffers[0])

    elapsed_time = 0
    
    # Benchmark with rotating buffers to flush L2 cache
    for i in range(num_iter):
        buf_idx = i % num_buffers
        A = A_buffers[buf_idx]
        B = B_buffers[buf_idx]
        C = C_buffers[buf_idx]
        torch.cuda.synchronize()
        start_event.record()
        gemm_func(A, B, C)
        if flush:
            flush(A,B,C)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time += start_event.elapsed_time(end_event)

    avg_elapsed_time = elapsed_time / num_iter
    tflops = int(flop / (avg_elapsed_time * 1e9))
    print(f"m={m},n={n},k={k}: {tflops} TFLOPS")