from setuptools import setup, find_packages

setup(
    name='pso_optimizer',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'joblib',
        'tqdm'
    ],
    author='Mert Bayraktar',
    author_email='mertbayraktar07@outlook.com',
    description='Hyperparameter selection on machine learning models using PSO algorithm.',
    url='https://github.com/mBayraktar12/PSO-Hyperparameter-Selection'
)
