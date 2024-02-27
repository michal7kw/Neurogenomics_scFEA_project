import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")

# If CUDA is available, print the number of CUDA devices and their names
if cuda_available:
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")


# choose cpu or gpu automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("now you are using device: ", device)
