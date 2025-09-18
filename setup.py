from setuptools import setup, find_packages

setup(
    name="autoformer-keras",
    version="0.1.0",
    description="Keras implementation of Autoformer for long-term time series forecasting",
    author="ksra7900",
    url="https://github.com/ksra7900/Autoformer-Keras",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
