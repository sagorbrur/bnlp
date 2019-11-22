import codecs
import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="bnlp",
    version="0.0.1",
    author="Sagor Sarker",
    author_email="sagorhem3532@gmail.com",
    description="A small example package",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
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
        "gensim>=3.7.3",
        "nltk",
    ],
)