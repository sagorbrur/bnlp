import codecs
import setuptools


setuptools.setup(
    name="bnlp_toolkit",
    version="4.0.3",
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
    python_requires=">=3.6",
    install_requires=[
        "sentencepiece==0.2.0",
        "gensim==4.3.2",
        "nltk",
        "numpy",
        "scipy==1.10.1",
        "sklearn-crfsuite==0.3.6",
        "tqdm==4.66.3",
        "ftfy==6.2.0",
        "emoji==1.7.0",
        "requests",
    ],
    extras_require={
        "fasttext": ["fasttext==0.9.2"],
    },
)
