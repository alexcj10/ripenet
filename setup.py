from setuptools import setup, find_packages

setup(
    name="ripenet",
    version="1.0.0",
    description="Advanced AI Fruit Quality Analysis Suite",
    author="ALEX",
    packages=find_packages(),
    py_modules=["ripenet-cli"],
    install_requires=[
        "requests",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "ripenet=ripenet-cli:main",
        ],
    },
    python_requires=">=3.7",
)
