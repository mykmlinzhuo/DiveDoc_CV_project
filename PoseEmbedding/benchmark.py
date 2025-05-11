import torch
import time
from typing import List, Dict, Optional, Union
import sys
import os

# Import the optimized encoder
from pose_embedding import FullyOptimizedHierarchicalEncoder, OptimizedHierarchicalEncoder, HierarchicalSkeletalEncoder, WeightedMLPBasedEncoder, NaiveMLPEncoder

# Path to the directory containing the original encoders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_test_data(batch_size=10, num_joints=17):
    """Generate test data for benchmarking."""
    detections = []
    for i in range(batch_size):
        detections.append({
            'keypoints': torch.rand(num_joints, 2) * 100,
            'scores': torch.rand(num_joints)
        })
    return detections

def benchmark_encoder(encoder, detections, num_iterations=10):
    """Benchmark an encoder with the given detections."""
    # Move encoder to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    encoder = encoder.to(device)
    
    # Move detections to the device
    for i, det in enumerate(detections):
        if det is not None:
            detections[i] = {
                'keypoints': det['keypoints'].to(device),
                'scores': det['scores'].to(device)
            }
    
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        print(f"Processing batch {_ + 1}/{num_iterations}")
        encoder(detections)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate test data
    print("Generating test data...")
    detections = generate_test_data(batch_size=100)
    
    # Initialize encoders
    print("Initializing encoders...")
    h_encoder = HierarchicalSkeletalEncoder()
    w_encoder = WeightedMLPBasedEncoder()
    n_encoder = NaiveMLPEncoder()
    o_encoder = OptimizedHierarchicalEncoder()
    f_encoder = FullyOptimizedHierarchicalEncoder()

    
    # Count parameters
    h_params = sum(p.numel() for p in h_encoder.parameters() if p.requires_grad)
    w_params = sum(p.numel() for p in w_encoder.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in n_encoder.parameters() if p.requires_grad)
    o_params = sum(p.numel() for p in o_encoder.parameters() if p.requires_grad)
    f_params = sum(p.numel() for p in f_encoder.parameters() if p.requires_grad)

    print(f"HierarchicalSkeletalEncoder parameters: {h_params}")
    print(f"WeightedMLPBasedEncoder parameters: {w_params}")
    print(f"NaiveMLPEncoder parameters: {n_params}")
    print(f"OptimizedHierarchicalEncoder parameters: {o_params}")
    print(f"FullyOptimizedHierarchicalEncoder parameters: {f_params}")
    
    # Benchmark encoders
    print("\nBenchmarking encoders...")
    
    print("Benchmarking HierarchicalSkeletalEncoder...")
    h_time = benchmark_encoder(h_encoder, detections)
    
    print("Benchmarking WeightedMLPBasedEncoder...")
    w_time = benchmark_encoder(w_encoder, detections)
    
    print("Benchmarking NaiveMLPEncoder...")
    n_time = benchmark_encoder(n_encoder, detections)
    
    print("Benchmarking OptimizedHierarchicalEncoder...")
    o_time = benchmark_encoder(o_encoder, detections)

    print("Benchmarking FullyOptimizedHierarchicalEncoder...")
    f_time = benchmark_encoder(f_encoder, detections)
    
    # Print results
    print("\nResults:")
    print(f"HierarchicalSkeletalEncoder: {h_time:.6f} sec per iteration")
    print(f"WeightedMLPBasedEncoder: {w_time:.6f} sec per iteration")
    print(f"NaiveMLPEncoder: {n_time:.6f} sec per iteration")
    print(f"OptimizedHierarchicalEncoder: {o_time:.6f} sec per iteration")
    print(f"FullyOptimizedHierarchicalEncoder: {f_time:.6f} sec per iteration")
    
    print(f"\nSpeedup (Original vs Optimized): {h_time/o_time:.2f}x")
    print(f"Speedup (Original vs Fully Optimized): {h_time/f_time:.2f}x")

if __name__ == "__main__":
    main()