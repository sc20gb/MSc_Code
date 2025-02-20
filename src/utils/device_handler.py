import torch

def handle_devices(cpu_only=False):
    """Manages device assignment for multi-GPU training.
    
    Assigns visual encoder and LLM to appropriate devices based on:
    - GPU availability 
    - Number of available GPUs
    - cpu_only flag
    
    Args:
        cpu_only (bool): Force CPU usage even if GPUs available
        
    Returns:
        tuple: (device_vit, device_llm) - Devices for visual encoder and LLM
    """
    if torch.cuda.is_available() and not cpu_only:
        gpu_count = torch.cuda.device_count()
        available_gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"CUDA is available with {gpu_count} GPU(s)!")
        print("Available GPUs:", ", ".join(available_gpu_names))
        
        if gpu_count >= 2:
            # Assign the first GPU for the visual encoder
            device_vit = torch.device("cuda:0")
            print(f"Visual encoder will run on GPU 0: {available_gpu_names[0]}")

            # Assign the second GPU for the connector LLM
            device_llm = torch.device("cuda:1")
            print(f"Connector LLM will run on GPU 1: {available_gpu_names[1]}")
        else:
            print("Only one GPU available, models are split between GPU 0")
            device_vit = torch.device("cuda:0")
            device_llm = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device_vit = torch.device("cpu")
        device_llm = torch.device("cpu")

    return device_vit, device_llm