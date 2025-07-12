import torch
import torch.nn as nn
from learn2com.nn_networks.C_Net import C_Net

def inspect_layer_weights(layer, layer_name="Layer"):
    """Inspect weights of a specific layer"""
    print(f"\n{'='*50}")
    print(f"INSPECTING {layer_name}")
    print(f"{'='*50}")
    
    print(f"Layer type: {type(layer)}")
    print(f"Layer structure:\n{layer}")
    
    # Get all parameters in this layer
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"\nTotal parameters in {layer_name}: {total_params:,}")
    
    print(f"\n{'Parameter Name':<20} {'Shape':<15} {'Parameters':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 90)
    
    for name, param in layer.named_parameters():
        shape_str = str(list(param.shape))
        param_count = param.numel()
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        
        print(f"{name:<20} {shape_str:<15} {param_count:<12} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    # Check for NaN/Inf
    print(f"\nChecking for NaN/Inf values:")
    has_issues = False
    for name, param in layer.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️  NaN found in {name}")
            has_issues = True
        if torch.isinf(param).any():
            print(f"⚠️  Inf found in {name}")
            has_issues = True
    
    if not has_issues:
        print("✅ No NaN or Inf values found")

def inspect_gru_layer(gru_layer, layer_name="GRU"):
    """Special inspection for GRU layers"""
    print(f"\n{'='*50}")
    print(f"INSPECTING {layer_name} LAYER")
    print(f"{'='*50}")
    
    # GRU specific parameters
    print(f"Input size: {gru_layer.input_size}")
    print(f"Hidden size: {gru_layer.hidden_size}")
    print(f"Number of layers: {gru_layer.num_layers}")
    print(f"Bidirectional: {gru_layer.bidirectional}")
    
    # Inspect each parameter
    for name, param in gru_layer.named_parameters():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Parameters: {param.numel():,}")
        print(f"  Mean: {param.mean():.6f}")
        print(f"  Std: {param.std():.6f}")
        print(f"  Min: {param.min():.6f}")
        print(f"  Max: {param.max():.6f}")
        
        # Show first few values
        if param.numel() <= 20:
            print(f"  Values: {param.flatten()}")
        else:
            print(f"  First 10 values: {param.flatten()[:10]}")
        
        # Check for extreme values
        if param.abs().max() > 10:
            print(f"  ⚠️  Large values detected (max abs: {param.abs().max():.2f})")

def inspect_linear_layer(linear_layer, layer_name="Linear"):
    """Special inspection for Linear layers"""
    print(f"\n{'='*50}")
    print(f"INSPECTING {layer_name} LAYER")
    print(f"{'='*50}")
    
    print(f"Input features: {linear_layer.in_features}")
    print(f"Output features: {linear_layer.out_features}")
    
    # Weight matrix
    weight = linear_layer.weight
    bias = linear_layer.bias
    
    print(f"\nWeight matrix:")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean: {weight.mean():.6f}")
    print(f"  Std: {weight.std():.6f}")
    print(f"  Min: {weight.min():.6f}")
    print(f"  Max: {weight.max():.6f}")
    
    # Show weight matrix heatmap (first 10x10 if large)
    if weight.shape[0] <= 10 and weight.shape[1] <= 10:
        print(f"  Weight matrix:\n{weight}")
    else:
        print(f"  Weight matrix (first 5x5):\n{weight[:5, :5]}")
    
    if bias is not None:
        print(f"\nBias vector:")
        print(f"  Shape: {bias.shape}")
        print(f"  Values: {bias}")

def show_weight_histogram(layer, layer_name="Layer"):
    """Show weight distribution"""
    print(f"\n{'='*50}")
    print(f"WEIGHT DISTRIBUTION FOR {layer_name}")
    print(f"{'='*50}")
    
    all_weights = []
    for name, param in layer.named_parameters():
        all_weights.extend(param.flatten().tolist())
    
    if all_weights:
        weights_tensor = torch.tensor(all_weights)
        print(f"All weights statistics:")
        print(f"  Mean: {weights_tensor.mean():.6f}")
        print(f"  Std: {weights_tensor.std():.6f}")
        print(f"  Min: {weights_tensor.min():.6f}")
        print(f"  Max: {weights_tensor.max():.6f}")
        print(f"  Median: {weights_tensor.median():.6f}")
        
        # Show histogram bins
        hist = torch.histc(weights_tensor, bins=10)
        print(f"  Histogram (10 bins): {hist.tolist()}")

# Example usage
if __name__ == "__main__":
    # Create network
    cnet = C_Net(obs_dims=1, number_of_agents=2, action_dims=5, message_dims=1, embedding_dim=128)
    
    # Inspect specific layers
    inspect_gru_layer(cnet.gru1, "GRU1")
    inspect_gru_layer(cnet.gru2, "GRU2")
    inspect_linear_layer(cnet.mlp_layer.fc1, "MLP FC1")
    inspect_linear_layer(cnet.mlp_layer.fc2, "MLP FC2")
    inspect_layer_weights(cnet.task_specific_net, "Task Specific Net")
    inspect_layer_weights(cnet.reciver, "Receiver")
    
    # Show weight distributions
    show_weight_histogram(cnet.gru1, "GRU1")
    show_weight_histogram(cnet.mlp_layer, "MLP Layer") 