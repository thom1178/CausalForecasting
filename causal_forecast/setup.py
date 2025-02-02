from setuptools import setup, find_packages

setup(
    name="causal_forecast",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "networkx>=2.5",
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.12.0",
    ],
    author="Jamel Thomas",
    author_email="contact@jamelt.com",
    description="A package for causal forecasting with what-if scenario analysis",
    long_description="A package for causal forecasting with what-if scenario analysis",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causal_forecast",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 