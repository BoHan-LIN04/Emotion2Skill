from setuptools import setup, find_packages

setup(
    name             = "emotion2skill",
    version          = "0.1.0",
    description      = "Emotion2Skill: Grounding Agent Skill Selection in LLM-Internal Emotion Vectors",
    python_requires  = ">=3.10",
    packages         = find_packages(),
    install_requires = [
        "torch>=2.1.0",
        "transformers>=4.43.0",
        "accelerate>=0.30.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require   = {
        "dev": ["pytest>=7.4.0", "matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)
