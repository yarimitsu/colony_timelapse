from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="colony_timelapse",
    version="1.0.0",
    author="Colony Timelapse Team",
    description="Process and analyze colony timelapse images using YOLO classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/colony_timelapse",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'process-year=scripts.process_year:main',
        ],
    },
) 