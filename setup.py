from setuptools import setup, find_packages

setup(
    name="SD-orb",
    version="0.1.0",
    description="A real-time AI VJ orchestrator using Stable Diffusion and TensorRT",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tensorrt",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "opencv-python-headless",
        "dearpygui",
        "pygame-ce",
        "PyAudio",
        "streamdiffusion",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "sd-orb=main:main",
        ],
    },
    python_requires=">=3.10",
)
