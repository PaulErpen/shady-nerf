from setuptools import setup, find_packages

setup(
    name="lodnelf",
    version="0.1",
    packages=find_packages(include=["lodnelf"]),
    test_suite="test_package",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.13.1",
        "numpy>=1.21.1",
        "matplotlib>=3.4.3",
        "pandas>=1.3.3",
        "imageio>=2.34.1",
        "scikit-image>=0.23.2",
        "h5py>=3.11.0",
        "tqdm==2.2.3",
        "wandb==0.17.0",
        "pygame==2.5.2",
        "einops==0.8.0",
    ],
)
