from setuptools import setup, find_packages

setup(
    name="datascope-ml",
    version="2.0.1",
    description="The official Python SDK for DataScope: The Machine Learning Observability Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Akshat Kankani",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pandas>=1.2.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
