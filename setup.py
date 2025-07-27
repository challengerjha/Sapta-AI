from setuptools import setup, find_packages

setup(
    name="sapta-ai",
    version="0.1.0",
    author="Challenger Jha",
    description="Sapta AI: Open Source AI Model",
    packages=find_packages(),  # Automatically finds 'sapta'
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
