import torch
import torch.nn as nn
from networks.C_Net import C_Net

def inspect_network_params(network, network_name="Network"):
    """Comprehensive network parameter inspection"""
    print(f"\n{'='*50}")
    print(f"INSPECTING {network_name}")
    print(f"{'='*50}")
    
    # Basic info
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Detailed parameter breakdown
    print(f"\n{'Layer Name':<30} {'Shape':<20} {'Parameters':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 100)
    
    for name, param in network.named_parameters():
        shape_str = str(list(param.shape))
        param_count = param.numel()
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        
        print(f"{name:<30} {shape_str:<20} {param_count:<12} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    # Check for NaN or Inf values
    print(f"\nChecking for NaN/Inf values:")
    has_nan = False
    has_inf = False
    
    for name, param in network.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️  NaN found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"⚠️  Inf found in {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("✅ No NaN or Inf values found")
    
    # Parameter gradients (if available)
    print(f"\nGradient information:")
    has_grad = False
    for name, param in network.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
    
    if not has_grad:
        print("No gradients available (run backward pass first)")

def compare_networks(net1, net2, name1="Network1", name2="Network2"):
    """Compare two networks"""
    print(f"\n{'='*60}")
    print(f"COMPARING {name1} vs {name2}")
    print(f"{'='*60}")
    
    params1 = dict(net1.named_parameters())
    params2 = dict(net2.named_parameters())
    
    for name in params1.keys():
        if name in params2:
            diff = torch.abs(params1[name] - params2[name]).mean().item()
            print(f"{name}: mean difference = {diff:.8f}")
        else:
            print(f"{name}: only in {name1}")
    
    for name in params2.keys():
        if name not in params1:
            print(f"{name}: only in {name2}")

def check_parameter_updates(network, name="Network"):
    """Check if parameters have been updated"""
    print(f"\n{'='*40}")
    print(f"CHECKING PARAMETER UPDATES FOR {name}")
    print(f"{'='*40}")
    
    # Store current parameters
    current_params = {}
    for name_param, param in network.named_parameters():
        current_params[name_param] = param.data.clone()
    
    return current_params

def detect_parameter_changes(network, old_params, name="Network"):
    """Detect parameter changes after training"""
    print(f"\n{'='*40}")
    print(f"DETECTING PARAMETER CHANGES FOR {name}")
    print(f"{'='*40}")
    
    for name_param, param in network.named_parameters():
        if name_param in old_params:
            change = torch.abs(param.data - old_params[name_param]).mean().item()
            print(f"{name_param}: mean change = {change:.8f}")

# Example usage
if __name__ == "__main__":
    # Create networks
    cnet = C_Net(obs_dims=1, number_of_agents=2, action_dims=5, message_dims=1, embedding_dim=128)
    cnet_target = C_Net(obs_dims=1, number_of_agents=2, action_dims=5, message_dims=1, embedding_dim=128)
    
    # Inspect main network
    inspect_network_params(cnet, "Main Network")
    
    # Compare networks
    compare_networks(cnet, cnet_target, "Main Network", "Target Network")
    
    # Store parameters before training
    old_params = check_parameter_updates(cnet, "Main Network")
    
    # Simulate some parameter changes
    with torch.no_grad():
        for param in cnet.parameters():
            param.data += torch.randn_like(param) * 0.01
    
    # Detect changes
    detect_parameter_changes(cnet, old_params, "Main Network") 