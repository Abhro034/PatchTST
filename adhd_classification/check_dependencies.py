"""
Check if required dependencies are installed
"""

import sys

def check_dependency(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def main():
    print("Checking dependencies for ADHD Classification Pipeline...")
    print("=" * 60)

    required = {
        'torch': 'torch',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
    }

    optional = {
        'mne': 'mne',
        'pandas': 'pandas',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
    }

    print("\nRequired dependencies:")
    all_required = True
    for package, import_name in required.items():
        if not check_dependency(package, import_name):
            all_required = False

    print("\nOptional dependencies (for FIF data and advanced features):")
    for package, import_name in optional.items():
        check_dependency(package, import_name)

    print("\n" + "=" * 60)

    if all_required:
        print("✓ All required dependencies are installed!")
        print("\nYou can now run:")
        print("  python examples/sanity_check.py")
        print("  python examples/train_baseline.py")
    else:
        print("✗ Some required dependencies are missing!")
        print("\nPlease install missing dependencies:")
        print("  pip install torch numpy scikit-learn matplotlib")
        print("\nFor FIF data support, also install:")
        print("  pip install mne pandas seaborn tqdm")

    return all_required


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
