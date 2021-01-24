from setuptools import setup
from setuptools import find_packages

setup(
    name="MAN",
    version="0.0.1",
    description="Multi-pretext Attention Network for Few-shot Learning with Self-supervision",
    packages=find_packages(),
    install_requires=["tensorboardX",
                      "tqdm",
                      "numpy",
                      "torch",
                      "torchvision",
                      "Pillow",
                      'torchnet'],
)
