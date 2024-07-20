import torch
total_memory = torch.cuda.get_device_properties(0).total_memory
print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
allocated_memory = torch.cuda.memory_allocated()
print(f"Currently allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
available_memory = total_memory - allocated_memory
print(f"Available GPU memory: {available_memory / 1e9:.2f} GB")