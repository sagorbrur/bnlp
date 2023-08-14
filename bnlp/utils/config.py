"""Module providing Config for model name and URL."""

class ModelTypeEnum:
    NER = "NER"
    POS = "POS"
    SENTENCEPIECE = "SPM"
    FASTTEXT = "FASTTEXT"
    GLOVE = "GLOVE"
    NEWS_DOC2VEC = "NEWS_DOC2VEC"
    WIKI_DOC2VEC = "WIKI_DOC2VEC"
    WORD2VEC = "WORD2VEC"


class ModelInfo():
    """Class for various model name and their URLs
    """
    __url_dict = {
        "NER" : {
            "name" : "bn_ner.pkl",
            "type" : "single",
            "url" : "https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_ner.pkl",
        },
        "POS" : {
            "name" : "bn_pos.pkl",
            "type" : "single",
            "url" : "https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_pos.pkl",
        },
        "SPM" : {
            "name" : "bn_spm.model",
            "type" : "single",
            "url" : "https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_spm.model",
        },
        "FASTTEXT" : {
            "name" : "bengali_fasttext_wiki.bin",
            "type" : "zip",
            "url" : "https://huggingface.co/sagorsarker/bangla-fasttext/resolve/main/bengali_fasttext_wiki.zip",
        },
        "GLOVE" : {
            "name" : "bn_glove.39M.100d.txt",
            "type" : "zip",
            "url" : "https://huggingface.co/sagorsarker/bangla-glove-vectors/resolve/main/bn_glove.39M.100d.zip",
        },
        "NEWS_DOC2VEC" : {
            "name" : "bangla_news_article_doc2vec.model",
            "type" : "zip",
            "url" : "https://huggingface.co/sagorsarker/news_article_doc2vec/resolve/main/news_article_doc2vec.zip",
        },
        "WIKI_DOC2VEC" : {
            "name" : "bnwiki_doc2vec.model",
            "type" : "zip",
            "url" : "https://huggingface.co/sagorsarker/bnwiki_doc2vec_model/resolve/main/bnwiki_doc2vec_model.zip",
        },
        "WORD2VEC" : {
            "name" : "bnwiki_word2vec.model",
            "type" : "zip",
            "url" : "https://huggingface.co/sagorsarker/bangla_word2vec/resolve/main/bangla_word2vec_gen4.zip",
        },
    }

    @staticmethod
    def get_model_info(name:str) -> tuple:
        """Get Filename of the model

        Args:
            name (str): Name of the model

        Raises:
            KeyError: KeyError if model name not in config

        Returns:
            tuple: tuple (model name, model type, model URL)
        """
        try:
            model_info = ModelInfo.__url_dict[name]
            file_name = model_info["name"]
            model_type = model_info["type"]
            model_url = model_info["url"]
            return (file_name, model_type, model_url)
        except KeyError as key_err:
            print(f"{name} model not found in the configuration")
            raise key_err

    @staticmethod
    def get_all_models() -> list:
        """Get keys of all models

        Args:

        Returns:
            list: list of model keys
        """
        all_model_keys = list(ModelInfo.__url_dict.keys())
        return all_model_keys


