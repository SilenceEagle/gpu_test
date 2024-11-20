import torch
import time


def list_gpus():
    if not torch.cuda.is_available():
        print("No CUDA-compatible devices found.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"  - Total Memory: {
                torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f
                } GB")
        print(
            f"  - CUDA Capability: {
                torch.cuda.get_device_properties(i).major
                }.{torch.cuda.get_device_properties(i).minor}")
        print(
            f"  - Multi-Processor Count: {
                torch.cuda.get_device_properties(i).multi_processor_count}")


def test_gpus():
    # 使用 GPU 的张量运算
    if torch.cuda.is_available():
        device = torch.device("cuda")
        a = torch.rand(10000, 10000, device=device)
        b = torch.rand(10000, 10000, device=device)

        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
        print(f"GPU computation time: {time.time() - start_time:.4f} seconds")

    # 使用 CPU 的张量运算
    device = torch.device("cpu")
    a = torch.rand(10000, 10000, device=device)
    b = torch.rand(10000, 10000, device=device)

    start_time = time.time()
    c = torch.matmul(a, b)
    print(f"CPU computation time: {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    list_gpus()
    test_gpus()
