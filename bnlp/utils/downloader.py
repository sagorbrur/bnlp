"""Module providing Function for downloading models."""

import os
import shutil
import requests
from tqdm.auto import tqdm

from bnlp.utils.config import ModelInfo

def _create_dirs(model_name: str) -> str:
    """Create directories for downloading models

    Args:
        model_name (str): Name of the model

    Returns:
        str: Absolute path where model can be downloaded
    """
    model_dir = os.path.join(os.path.expanduser("~"), "bnlp", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    return model_path

def _download_file(file_url: str, file_path: str) -> str:
    """Function to download file

    Args:
        file_url (str): URL of the file
        file_path (str): Path where file will be downloaded

    Raises:
        network_error: Download related error

    Returns:
        str: Path where the file is downloaded
    """
    if os.path.exists(file_path):
        return file_path
    op_desc = f"Downloading {os.path.basename(file_path)}"
    try:
        with requests.Session() as req_sess:
            req_res = req_sess.get(file_url, stream=True)
            total_length = int(req_res.headers.get("Content-Length"))
            with tqdm.wrapattr(req_res.raw, "read", total=total_length, desc=op_desc) as raw:
                with open(file_path , "wb") as file:
                    shutil.copyfileobj(raw,file)
        return file_path
    except Exception as network_error:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise network_error

def download_model(name: str) -> str:
    """Download and extract model if necessary

    Args:
        name (str): _description_

    Returns:
        str: _description_
    """
    model_name, model_type, model_url = ModelInfo.get_model_info(name)
    model_path = _create_dirs(model_name)
    if model_type == "single":
        model_path = _download_file(model_url, model_path)
    else:
        print(f"model type {model_type} not yet implemented")
        model_path = ""
    return model_path

def download_all_models() -> None:
    """Download and extract all available models for BNLP
    """
    model_keys = ModelInfo.get_all_models()
    for model_key in model_keys:
        download_model(model_key)
