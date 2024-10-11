import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="strategic-ml",
    version="0.1.0",
    author="Your Name",
    author_email="orimintz21@gmail.com",
    description="A comprehensive library for strategic machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orimintz21/strategic_ml",
    project_urls={
        "Documentation": "https://github.com/orimintz21/strategic_ml#readme",
        "Source Code": "https://github.com/orimintz21/strategic_ml",
        "Bug Tracker": "https://github.com/orimintz21/strategic_ml/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",  # Adjust as appropriate
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",  
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.6",
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2",
        "numpy>=1.19.0",
    ],
    include_package_data=True,
)
