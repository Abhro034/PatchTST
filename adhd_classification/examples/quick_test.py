"""
Quick sanity test - just tests forward pass without training
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np

from modules.model import ADHDClassifier
from utils.config import ModelConfig

def test_model_step(step_name, config, device):
    """Test a single model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {step_name}")
    print(f"{'='*60}")

    try:
        # Create model
        model = ADHDClassifier(**config.to_dict()).to(device)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.get_num_params():,}")

        # Create dummy input [B, E, N, L]
        B, E, N, L = 2, 50, 10, 3000
        X = torch.randn(B, E, N, L).to(device)
        print(f"  Input shape: {X.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(X, return_intermediates=True)

        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {output['logits'].shape}")

        # Check output shape
        assert output['logits'].shape == (B, 1), f"Wrong output shape: {output['logits'].shape}"
        print(f"✓ Output shape correct")

        # Check for NaN/Inf
        has_nan = torch.isnan(output['logits']).any()
        has_inf = torch.isinf(output['logits']).any()

        if has_nan or has_inf:
            print(f"✗ NaN or Inf detected in output")
            return False

        print(f"✓ No numerical issues")

        # Print intermediate shapes
        print(f"\n  Intermediate tensors:")
        for key, value in output['intermediates'].items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")

        print(f"\n✓ {step_name} PASSED")
        return True

    except Exception as e:
        print(f"✗ {step_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("ADHD Classification Pipeline - Quick Sanity Test")
    print("="*60)

    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Test each step
    results = {}

    # Step 1: TCN Baseline
    print("\n" + "="*60)
    print("STEP 1: TCN Baseline")
    print("="*60)
    config1 = ModelConfig.step1_tcn_baseline()
    config1.seq_len = 3000
    results['Step 1'] = test_model_step("Step 1: TCN Baseline", config1, device)

    # Step 2: PatchTST
    print("\n" + "="*60)
    print("STEP 2: PatchTST Encoder")
    print("="*60)
    config2 = ModelConfig.step2_patchtst(seq_len=3000)
    results['Step 2'] = test_model_step("Step 2: PatchTST", config2, device)

    # Step 3: Tokens
    print("\n" + "="*60)
    print("STEP 3: Token Outputs")
    print("="*60)
    config3 = ModelConfig.step3_tokens(seq_len=3000)
    results['Step 3'] = test_model_step("Step 3: Tokens", config3, device)

    # Step 4: Graph
    print("\n" + "="*60)
    print("STEP 4: Graph Neural Networks")
    print("="*60)
    config4 = ModelConfig.step4_graph(seq_len=3000)
    results['Step 4'] = test_model_step("Step 4: Graph", config4, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for step, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{step}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All steps passed! Model architecture is working correctly.")
    else:
        print("\n✗ Some steps failed. Please review errors above.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
