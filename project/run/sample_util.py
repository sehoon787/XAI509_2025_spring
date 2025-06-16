# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard library imports
import glob
import io
import os
from typing import Dict
import re

# Third-party imports
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

# 전역 리스트: 전처리에서 제외된 샘플들의 UID 기록
skipped_uids = []

# 비음성 메타태그 제거 함수
def clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text).strip()  # 메타태그 제거
    text = re.sub(r"[^A-Z |]", "", text.upper())    # 허용된 문자만 유지 (A~Z, 공백, |)
    return text

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
    try:
        # Load waveform from raw bytes
        waveform, sample_rate = torchaudio.load(io.BytesIO(sample["wav"]))

        # Mono channel conversion: average if multichannel
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze()

        # 너무 짧은 오디오는 학습 불가능하므로 필터링
        if waveform.numel() < sample_rate * 0.3:  # 약 0.3초 미만
            raise ValueError(f"Audio too short: {waveform.numel()} samples")

        # 텍스트 디코딩 및 대문자 변환
        raw_text = sample["txt"].decode("utf-8").upper()

        '''
        라벨 텍스트가 소문자 등 tokenizer의 vocab에 없는 문자를 포함해 대부분 <unk>로 처리되었기 때문에, 이를 방지하기 위해 대문자로 변환
        '''
        # 메타태그 제거
        text = clean_text(raw_text)

        # 메타태그 제거 후도 텍스트가 비어 있다면 필터링
        if len(text) < 1:
            raise ValueError("Transcript empty after removing nonverbal tags")

        # Tokenize
        labels = processor.tokenizer(text).input_ids
        if processor.tokenizer.unk_token_id in labels:
            raise ValueError("UNK token detected in label")

        # Feature extraction (e.g., Wav2Vec2 input)
        input_values = processor.feature_extractor(
            waveform, sampling_rate=sample_rate
        ).input_values[0]

        return {"input_values": input_values, "labels": labels}

    except Exception as e:
        # sample에 '__key__'가 있다면 해당 UID 기록
        if "__key__" in sample:
            skipped_uids.append(sample["__key__"])
        return None


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
        .to_tuple("wav", "txt", "__key__")
        .map(lambda sample: {"wav": sample[0], "txt": sample[1], "__key__": sample[2]})
        .map(preprocess_sample)
        .select(lambda x: x is not None)
    )

    # 전처리 후 제외된 UID들을 파일로 저장
    if skipped_uids:
        with open("skipped_samples.txt", "w", encoding="utf-8") as f:
            for uid in skipped_uids:
                f.write(uid + "\n")
        print(f"[INFO] 제외된 샘플 {len(skipped_uids)}개를 skipped_samples.txt에 저장했습니다.")

    return dataset
