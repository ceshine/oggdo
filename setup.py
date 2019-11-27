from setuptools import setup, find_packages

from oggdo.version import __version__

setup(
    name="oggdo",
    version=__version__,
    author="Ceshine Lee",
    author_email="ceshine@ceshine.net",
    description="Sentence Embeddings using BERT",
    license="GLWT(Good Luck With That)",
    url="https://github.com/ceshine/oggdo",
    packages=find_packages(exclude=['scripts']),
    install_requires=[
        "transformers>=2.1.1",
        "tqdm",
        "torch>=1.3.0",
        "numpy",
        "scikit-learn",
        "scipy"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer BERT sentence embedding PyTorch NLP deep learning"
)
