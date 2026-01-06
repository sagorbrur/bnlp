#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BNLP Command Line Interface

A command-line tool for Bengali Natural Language Processing.

Usage:
    bnlp tokenize <text> [--type=<tokenizer>]
    bnlp ner <text> [--model=<path>]
    bnlp pos <text> [--model=<path>]
    bnlp embedding <word> [--type=<embedding>] [--model=<path>]
    bnlp clean <text> [--options...]
    bnlp download <model> [--all]
    bnlp list-models
    bnlp --version
    bnlp --help
"""

import argparse
import json
import sys
from typing import List, Optional

from bnlp import __version__


def tokenize_command(args: argparse.Namespace) -> None:
    """Handle tokenize command."""
    text = args.text
    tokenizer_type = args.type.lower()

    if tokenizer_type == "basic":
        from bnlp import BasicTokenizer
        tokenizer = BasicTokenizer()
        tokens = tokenizer.tokenize(text)
    elif tokenizer_type == "nltk":
        from bnlp import NLTKTokenizer
        tokenizer = NLTKTokenizer()
        if args.sentence:
            tokens = tokenizer.sentence_tokenize(text)
        else:
            tokens = tokenizer.word_tokenize(text)
    elif tokenizer_type == "sentencepiece":
        from bnlp import SentencepieceTokenizer
        tokenizer = SentencepieceTokenizer(model_path=args.model or "")
        tokens = tokenizer.tokenize(text)
    else:
        print(f"Error: Unknown tokenizer type '{tokenizer_type}'", file=sys.stderr)
        print("Available types: basic, nltk, sentencepiece", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(tokens, ensure_ascii=False))
    else:
        print(tokens)


def ner_command(args: argparse.Namespace) -> None:
    """Handle NER command."""
    from bnlp import BengaliNER

    ner = BengaliNER(model_path=args.model or "")
    result = ner.tag(args.text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result)


def pos_command(args: argparse.Namespace) -> None:
    """Handle POS tagging command."""
    from bnlp import BengaliPOS

    pos = BengaliPOS(model_path=args.model or "")
    result = pos.tag(args.text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result)


def embedding_command(args: argparse.Namespace) -> None:
    """Handle word embedding command."""
    word = args.word
    embedding_type = args.type.lower()

    if embedding_type == "word2vec":
        from bnlp import BengaliWord2Vec
        model = BengaliWord2Vec(model_path=args.model or "")
        if args.similar:
            result = model.get_most_similar_words(word, topn=args.topn)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(f"Most similar words to '{word}':")
                for w, score in result:
                    print(f"  {w}: {score:.4f}")
        else:
            vector = model.get_word_vector(word)
            if args.json:
                print(json.dumps(vector.tolist()))
            else:
                print(f"Vector for '{word}' (shape: {vector.shape}):")
                print(vector)
    elif embedding_type == "fasttext":
        try:
            from bnlp.embedding.fasttext import BengaliFasttext
            model = BengaliFasttext(model_path=args.model or "")
            vector = model.get_word_vector(word)
            if args.json:
                print(json.dumps(vector.tolist()))
            else:
                print(f"Vector for '{word}' (shape: {vector.shape}):")
                print(vector)
        except ImportError:
            print("Error: fasttext not installed. Install with: pip install fasttext", file=sys.stderr)
            sys.exit(1)
    elif embedding_type == "glove":
        from bnlp import BengaliGlove
        model = BengaliGlove(glove_vector_path=args.model or "")
        if args.similar:
            result = model.get_closest_word(word)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(f"Closest words to '{word}':")
                for w in result:
                    print(f"  {w}")
        else:
            vector = model.get_word_vector(word)
            if args.json:
                print(json.dumps(vector.tolist()))
            else:
                print(f"Vector for '{word}' (shape: {vector.shape}):")
                print(vector)
    else:
        print(f"Error: Unknown embedding type '{embedding_type}'", file=sys.stderr)
        print("Available types: word2vec, fasttext, glove", file=sys.stderr)
        sys.exit(1)


def clean_command(args: argparse.Namespace) -> None:
    """Handle text cleaning command."""
    from bnlp import CleanText

    cleaner = CleanText(
        fix_unicode=args.fix_unicode,
        unicode_norm=args.unicode_norm,
        remove_url=args.remove_url,
        remove_email=args.remove_email,
        remove_number=args.remove_number,
        remove_emoji=args.remove_emoji,
        remove_punct=args.remove_punct,
    )
    result = cleaner(args.text)
    print(result)


def download_command(args: argparse.Namespace) -> None:
    """Handle model download command."""
    from bnlp.utils.downloader import download_model, download_all_models
    from bnlp.utils.config import ModelInfo

    if args.model.lower() == "all":
        print("Downloading all models...")
        download_all_models()
        print("All models downloaded successfully!")
    else:
        model_name = args.model.upper()
        available_models = ModelInfo.get_all_models()

        if model_name not in available_models:
            print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
            print(f"Available models: {', '.join(available_models)}", file=sys.stderr)
            sys.exit(1)

        print(f"Downloading {model_name} model...")
        path = download_model(model_name)
        print(f"Model downloaded to: {path}")


def list_models_command(args: argparse.Namespace) -> None:
    """List all available models."""
    from bnlp.utils.config import ModelInfo

    models = ModelInfo.get_all_models()
    print("Available models:")
    print("-" * 40)
    for model in models:
        info = ModelInfo.get_model_info(model)
        print(f"  {model}")
        print(f"    File: {info[0]}")
        print(f"    Type: {info[1]}")
        print()


def corpus_command(args: argparse.Namespace) -> None:
    """Handle corpus information command."""
    from bnlp import BengaliCorpus

    resource = args.resource.lower()

    if resource == "stopwords":
        if args.json:
            print(json.dumps(BengaliCorpus.stopwords, ensure_ascii=False))
        else:
            print(f"Bengali Stopwords ({len(BengaliCorpus.stopwords)} words):")
            print(", ".join(BengaliCorpus.stopwords[:20]) + "...")
    elif resource == "letters":
        print(f"Bengali Letters: {BengaliCorpus.letters}")
    elif resource == "digits":
        print(f"Bengali Digits: {BengaliCorpus.digits}")
    elif resource == "vowels":
        print(f"Bengali Vowels: {BengaliCorpus.vowels}")
    elif resource == "punctuations":
        print(f"Bengali Punctuations: {BengaliCorpus.punctuations}")
    else:
        print(f"Error: Unknown resource '{resource}'", file=sys.stderr)
        print("Available resources: stopwords, letters, digits, vowels, punctuations", file=sys.stderr)
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bnlp",
        description="BNLP: Bengali Natural Language Processing Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bnlp tokenize "আমি বাংলায় গান গাই।"
  bnlp tokenize "আমি বাংলায় গান গাই।" --type nltk
  bnlp ner "সজীব ওয়াজেদ জয় ঢাকায় থাকেন।"
  bnlp pos "আমি ভাত খাই।"
  bnlp embedding "বাংলা" --similar
  bnlp clean "hello@example.com আমি বাংলায়" --remove-email
  bnlp download all
  bnlp download word2vec
  bnlp list-models
  bnlp corpus stopwords
        """
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"bnlp {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize Bengali text")
    tokenize_parser.add_argument("text", help="Text to tokenize")
    tokenize_parser.add_argument(
        "--type", "-t",
        default="basic",
        choices=["basic", "nltk", "sentencepiece"],
        help="Tokenizer type (default: basic)"
    )
    tokenize_parser.add_argument(
        "--sentence", "-s",
        action="store_true",
        help="Sentence tokenization (only for nltk)"
    )
    tokenize_parser.add_argument(
        "--model", "-m",
        help="Path to custom model (for sentencepiece)"
    )
    tokenize_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    tokenize_parser.set_defaults(func=tokenize_command)

    # NER command
    ner_parser = subparsers.add_parser("ner", help="Named Entity Recognition")
    ner_parser.add_argument("text", help="Text for NER")
    ner_parser.add_argument("--model", "-m", help="Path to custom NER model")
    ner_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    ner_parser.set_defaults(func=ner_command)

    # POS command
    pos_parser = subparsers.add_parser("pos", help="Part-of-Speech tagging")
    pos_parser.add_argument("text", help="Text for POS tagging")
    pos_parser.add_argument("--model", "-m", help="Path to custom POS model")
    pos_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    pos_parser.set_defaults(func=pos_command)

    # Embedding command
    embedding_parser = subparsers.add_parser("embedding", help="Word embeddings")
    embedding_parser.add_argument("word", help="Word to get embedding for")
    embedding_parser.add_argument(
        "--type", "-t",
        default="word2vec",
        choices=["word2vec", "fasttext", "glove"],
        help="Embedding type (default: word2vec)"
    )
    embedding_parser.add_argument("--model", "-m", help="Path to custom model")
    embedding_parser.add_argument(
        "--similar", "-s",
        action="store_true",
        help="Get similar words instead of vector"
    )
    embedding_parser.add_argument(
        "--topn", "-n",
        type=int,
        default=10,
        help="Number of similar words (default: 10)"
    )
    embedding_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    embedding_parser.set_defaults(func=embedding_command)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean and normalize text")
    clean_parser.add_argument("text", help="Text to clean")
    clean_parser.add_argument("--fix-unicode", action="store_true", default=True, help="Fix unicode (default: True)")
    clean_parser.add_argument("--unicode-norm", action="store_true", default=True, help="Normalize unicode (default: True)")
    clean_parser.add_argument("--remove-url", action="store_true", help="Remove URLs")
    clean_parser.add_argument("--remove-email", action="store_true", help="Remove emails")
    clean_parser.add_argument("--remove-number", action="store_true", help="Remove numbers")
    clean_parser.add_argument("--remove-emoji", action="store_true", help="Remove emojis")
    clean_parser.add_argument("--remove-punct", action="store_true", help="Remove punctuation")
    clean_parser.set_defaults(func=clean_command)

    # Download command
    download_parser = subparsers.add_parser("download", help="Download pre-trained models")
    download_parser.add_argument(
        "model",
        help="Model to download (or 'all' for all models)"
    )
    download_parser.set_defaults(func=download_command)

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    list_parser.set_defaults(func=list_models_command)

    # Corpus command
    corpus_parser = subparsers.add_parser("corpus", help="Access Bengali corpus data")
    corpus_parser.add_argument(
        "resource",
        choices=["stopwords", "letters", "digits", "vowels", "punctuations"],
        help="Corpus resource to display"
    )
    corpus_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    corpus_parser.set_defaults(func=corpus_command)

    return parser


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        parsed_args.func(parsed_args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
