#!/usr/bin/env python
"""
Transfer weights from 80-class COCO model to 2-class model (person, sports ball).

This script extracts weights for person (class 0) and sports ball (class 32) 
from a pretrained COCO model and creates a new checkpoint for 2-class detection.
"""

import argparse
import torch


def transfer_weights(src_checkpoint_path, dst_checkpoint_path):
    """
    Transfer weights from 80-class COCO model to 2-class model.
    
    Args:
        src_checkpoint_path: Path to 80-class COCO checkpoint
        dst_checkpoint_path: Path to save 2-class checkpoint
    """
    # Load source checkpoint
    print(f"Loading checkpoint from {src_checkpoint_path}")
    try:
        # PyTorch 2.6+ recommends weights_only=True for security
        checkpoint = torch.load(src_checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(src_checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Create new state dict
    new_state_dict = {}
    
    # COCO class indices for our target classes
    # person: class 0, sports ball: class 32
    src_indices = [0, 32]  # indices in 80-class model
    dst_indices = [0, 1]   # indices in 2-class model
    
    # Process each key in state dict
    for key, value in state_dict.items():
        # Handle classification layers
        if 'multi_level_conv_cls' in key:
            print(f"Processing classification layer: {key}")
            
            if key.endswith('.weight'):
                # Weight shape: [80, in_channels, h, w] -> [2, in_channels, h, w]
                if value.shape[0] == 80:
                    new_weight = torch.zeros(2, *value.shape[1:])
                    for dst_idx, src_idx in zip(dst_indices, src_indices):
                        new_weight[dst_idx] = value[src_idx]
                    new_state_dict[key] = new_weight
                    print(f"  Transferred weights for classes {src_indices} -> {dst_indices}")
                else:
                    new_state_dict[key] = value
                    
            elif key.endswith('.bias'):
                # Bias shape: [80] -> [2]
                if value.shape[0] == 80:
                    new_bias = torch.zeros(2)
                    for dst_idx, src_idx in zip(dst_indices, src_indices):
                        new_bias[dst_idx] = value[src_idx]
                    new_state_dict[key] = new_bias
                    print(f"  Transferred bias for classes {src_indices} -> {dst_indices}")
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
                
        # Keep all other layers unchanged
        else:
            new_state_dict[key] = value
    
    # Create new checkpoint
    new_checkpoint = {
        'state_dict': new_state_dict,
        'meta': checkpoint.get('meta', {}),
    }
    
    # Update meta info
    if 'meta' in new_checkpoint:
        new_checkpoint['meta']['num_classes'] = 2
        new_checkpoint['meta']['classes'] = ['person', 'sports ball']
        new_checkpoint['meta']['note'] = 'Transferred from 80-class COCO to 2-class (person, sports ball)'
    
    # Save checkpoint
    print(f"Saving 2-class checkpoint to {dst_checkpoint_path}")
    torch.save(new_checkpoint, dst_checkpoint_path)
    
    # Print summary
    print("\n=== Transfer Summary ===")
    print("Source: 80-class COCO model")
    print("Target: 2-class model (person, sports ball)")
    print(f"Transferred class indices: {src_indices} -> {dst_indices}")
    print(f"Output saved to: {dst_checkpoint_path}")
    
    # Verify shapes
    cls_keys = [k for k in new_state_dict.keys() if 'multi_level_conv_cls' in k]
    if cls_keys:
        sample_key = cls_keys[0]
        if sample_key.endswith('.weight'):
            print(f"Classification layer shape: {new_state_dict[sample_key].shape}")
        elif sample_key.endswith('.bias'):
            print(f"Classification bias shape: {new_state_dict[sample_key].shape}")
    
    return new_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Transfer COCO 80-class weights to 2-class model')
    parser.add_argument(
        'src_checkpoint',
        help='Path to source 80-class COCO checkpoint')
    parser.add_argument(
        'dst_checkpoint',
        help='Path to save 2-class checkpoint')
    
    args = parser.parse_args()
    
    transfer_weights(args.src_checkpoint, args.dst_checkpoint)


if __name__ == '__main__':
    main()