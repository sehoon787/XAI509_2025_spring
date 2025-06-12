# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
from typing import Dict

# Third-party imports
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    waveform, sample_rate = torchaudio.load(io.BytesIO(sample["wav"]))
    input_values = processor.feature_extractor(
        waveform[0], sampling_rate=sample_rate
    ).input_values[0]

    text = sample["txt"].decode("utf-8")
    labels = processor.tokenizer(text).input_ids

    return {"input_values": input_values, "labels": labels}


def make_dataset(data_dir: str) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    # Run on Windows OS
    shard_paths = glob.glob(os.path.join(data_dir, "shard-*.tar"))
    shard_urls = [f"file:{os.path.abspath(p).replace(os.sep, '/')}" for p in shard_paths]

    dataset = (
        wds.WebDataset(shard_urls)
        # wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
        .to_tuple("wav", "txt")
        .map(lambda sample: {"wav": sample[0], "txt": sample[1]})
        .map(preprocess_sample)
    )
    return dataset
