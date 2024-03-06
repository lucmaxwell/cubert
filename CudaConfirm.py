import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")
