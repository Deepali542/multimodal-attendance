from setuptools import setup, find_packages

setup(
    name="multimodal-attendance-engagement-agent",
    version="1.0.0",
    description="Privacy-preserving, offline-capable classroom attendance and engagement intelligence agent",
    author="Classroom AI Innovators",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.8.0.76",
        "streamlit>=1.30.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.1"
    ],
    entry_points={
        "console_scripts": [
            "classroom-agent=main:main",
        ],
    },
)
