from setuptools import setup, find_packages

setup(
    name="music_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "grpcio",
        "grpcio-tools",
        "joblib",
        "librosa",
        "numpy",
        "pandas",
    ],
)
