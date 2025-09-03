import torch

def run_gemm():
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = torch.matmul(a, b)
    print('GEMM test passed. Result shape:', c.shape)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA is not available!')
        exit(1)
    run_gemm()
