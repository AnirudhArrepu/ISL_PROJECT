from setuptools import setup, find_packages

setup(
    name="ISL-PROJECT",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},   # tells setuptools to look in src/
    install_requires=[
        "numpy",
        "pandas",
        "gymnasium",
        "mediapipe==0.10.21",
        "pybullet",
        "torch"
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "run-pipeline=main:run_pipeline",  # path inside package
        ],
    },
)
