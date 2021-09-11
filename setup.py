import codecs
import setuptools


setuptools.setup(
    name="bnlp_toolkit",
    version="3.1.2",
    author="Sagor Sarker",
    author_email="sagorhem3532@gmail.com",
    description="BNLP is a natural language processing toolkit for Bengali Language",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sagorbrur/bnlp",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "sentencepiece",
        "gensim==4.0.1",
        "nltk",
        "numpy",
        "scipy",
        "sklearn-crfsuite",
        "wasabi",
    ],
)
