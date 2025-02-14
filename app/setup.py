from setuptools import setup, find_packages

setup(
    name="Next Word Prediction",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit',
        'tensorflow',
        'numpy',
        'joblib',
    ],
)