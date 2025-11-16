"""
Simple code structure test - verifies imports and basic structure.
Doesn't require PyTorch - just checks the code is valid.

Usage:
    python tests/test_code_structure.py
"""

import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_vae_methods():
    """Check if VAE has required methods for diffusion model"""
    vae_file = project_root / "models" / "vae.py"

    with open(vae_file, 'r') as f:
        content = f.read()

    required_methods = [
        'def encode(',
        'def decode(',
        'def forward(',
    ]

    missing = []
    for method in required_methods:
        if method not in content:
            missing.append(method)

    return len(missing) == 0, missing


def check_train_vae_structure():
    """Check train_vae.py structure"""
    train_file = project_root / "train_vae.py"

    with open(train_file, 'r') as f:
        content = f.read()

    required_components = [
        'class AutoencoderLoss',
        'class VAETrainer',
        'def train_epoch(',
        'def validate(',
        'def main(',
    ]

    missing = []
    for component in required_components:
        if component not in content:
            missing.append(component)

    return len(missing) == 0, missing


def check_config_exists():
    """Check if VAE training config exists"""
    config_file = project_root / "config" / "vae_training.yaml"
    return config_file.exists()


def test_code_structure():
    """Run all structure tests"""

    print("\n" + "="*70)
    print("CODE STRUCTURE TEST")
    print("Verifying VAE training code structure (no PyTorch required)")
    print("="*70 + "\n")

    all_passed = True

    # Test 1: Syntax check for key files
    print("Test 1: Python Syntax Check")
    print("-" * 70)

    files_to_check = [
        "models/vae.py",
        "train_vae.py",
        "tests/test_vae_compatibility.py",
    ]

    for filepath in files_to_check:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"  {filepath}: ✗ NOT FOUND")
            all_passed = False
            continue

        valid, error = check_file_syntax(full_path)
        if valid:
            print(f"  {filepath}: ✓")
        else:
            print(f"  {filepath}: ✗ SYNTAX ERROR")
            print(f"    {error}")
            all_passed = False

    print()

    # Test 2: VAE methods
    print("Test 2: VAE Methods Check")
    print("-" * 70)

    has_methods, missing = check_vae_methods()
    if has_methods:
        print("  ✓ All required methods found")
        print("    - encode()")
        print("    - decode()")
        print("    - forward()")
    else:
        print(f"  ✗ Missing methods: {missing}")
        all_passed = False

    print()

    # Test 3: train_vae.py structure
    print("Test 3: Training Script Structure")
    print("-" * 70)

    has_components, missing = check_train_vae_structure()
    if has_components:
        print("  ✓ All required components found")
        print("    - AutoencoderLoss class")
        print("    - VAETrainer class")
        print("    - train_epoch() method")
        print("    - validate() method")
        print("    - main() function")
    else:
        print(f"  ✗ Missing components: {missing}")
        all_passed = False

    print()

    # Test 4: Config file
    print("Test 4: Configuration File")
    print("-" * 70)

    if check_config_exists():
        print("  ✓ config/vae_training.yaml exists")
    else:
        print("  ✗ config/vae_training.yaml NOT FOUND")
        all_passed = False

    print()

    # Test 5: Dataset parameter check
    print("Test 5: Dataset Parameters Check")
    print("-" * 70)

    train_file = project_root / "train_vae.py"
    with open(train_file, 'r') as f:
        content = f.read()

    # Check for correct parameters
    correct_params = [
        "data_dir=",      # Not dataset_path
        "val_ratio=",     # Not val_split
        "test_ratio=",    # Not test_split
    ]

    incorrect_params = [
        "dataset_path=",
        "val_split=",
        "test_split=",
    ]

    all_correct = all(param in content for param in correct_params)
    none_incorrect = all(param not in content for param in incorrect_params)

    if all_correct and none_incorrect:
        print("  ✓ Dataset parameters correct")
        print("    - Uses data_dir (not dataset_path)")
        print("    - Uses val_ratio/test_ratio (not val_split/test_split)")
    else:
        print("  ✗ Dataset parameters incorrect")
        if not all_correct:
            print(f"    Missing: {[p for p in correct_params if p not in content]}")
        if not none_incorrect:
            print(f"    Found incorrect: {[p for p in incorrect_params if p in content]}")
        all_passed = False

    print()

    # Test 6: Input shape handling
    print("Test 6: Input Shape Handling")
    print("-" * 70)

    # Check that we don't add extra unsqueeze(1)
    if "thick_slices.unsqueeze(1)" in content:
        print("  ✗ FAILED: Found thick_slices.unsqueeze(1)")
        print("    Dataset already returns (B, C, D, H, W)")
        all_passed = False
    else:
        print("  ✓ Correct: No extra unsqueeze(1)")
        print("    Uses dataset output directly: (B, C, D, H, W)")

    print()

    # Summary
    print("="*70)
    if all_passed:
        print("ALL STRUCTURE TESTS PASSED ✓")
        print("\nCode structure is correct and ready for VAE training")
    else:
        print("SOME TESTS FAILED ✗")
        print("\nPlease fix the issues above before proceeding")

    print("="*70 + "\n")

    return all_passed


if __name__ == '__main__':
    success = test_code_structure()
    sys.exit(0 if success else 1)
