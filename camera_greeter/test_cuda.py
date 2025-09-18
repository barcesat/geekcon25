import cv2
import numpy as np
import time

def test_cuda_availability():
    # Check if CUDA is available in the OpenCV build
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA-enabled GPU devices available: {cuda_available}")
    
    if cuda_available:
        # Get info about the CUDA device
        device_info = cv2.cuda.getDevice()
        print(f"Using CUDA device: {device_info}")
        
        # Get detailed device properties
        properties = cv2.cuda.DeviceInfo()
        print(f"Device name: {properties.name()}")
        print(f"Total memory: {properties.totalMemory() / (1024**2):.2f} MB")
        print(f"Multi Processor Count: {properties.multiProcessorCount()}")
        
        # Check if device is compatible
        print(f"CUDA device is compatible: {properties.isCompatible()}")
        
        # Test performance with a simple operation
        print("\nTesting performance with Gaussian Blur:")
        
        # Create a sample image
        img = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        
        # CPU version
        start_time = time.time()
        cpu_result = cv2.GaussianBlur(img, (31, 31), 0)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU version
        start_time = time.time()
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(img)
        gpu_result = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (31, 31), 0).apply(gpu_img)
        result_img = gpu_result.download()
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU is {speedup:.2f}x faster than CPU")
        
        return True
    else:
        print("CUDA is not available in your OpenCV build.")
        print("Please make sure you have installed OpenCV with CUDA support.")
        return False

if __name__ == "__main__":
    test_cuda_availability()