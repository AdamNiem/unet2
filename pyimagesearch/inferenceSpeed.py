
def measure_inference_time(model, input_tensor, num_warmup=50, num_runs=200):
    """
    Measures the inference time of a PyTorch model using CUDA events.

    Parameters:
    - model: the PyTorch model to evaluate.
    - input_tensor: the input tensor to feed to the model.
    - num_warmup: number of warm-up runs before measurements.
    - num_runs: number of timed runs for averaging inference time.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    # Warm-up runs
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Timing inference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    durations = []
    memory_stats = []
    with torch.inference_mode():
        for _ in range(num_runs):
            input_tensor = torch.randn(1, 3, 1216, 1920)  # Example input tensor
            
            input_tensor = input_tensor.to(device)
            start_event.record()
            outputs = model(input_tensor) 
            outputs = resize(outputs, size=[1200,1920])
            _, pred = torch.max(outputs, 1)
           # pred = pred.byte().cpu()
            end_event.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # memory usage
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert bytes to megabytes
            memory_stats.append(peak_memory)  # Store the peak memory usage

            # Measures time
            duration = start_event.elapsed_time(end_event)
            durations.append(duration)


    average_memory = sum(memory_stats) / len(memory_stats)
    print(f'Average peak memory {average_memory:.2f} MB')
    avg_duration = sum(durations) / len(durations)
    print(f"Average inference time: {avg_duration} ms")
