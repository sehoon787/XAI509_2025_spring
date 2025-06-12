# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import pipeline

# Custom imports
import sample_util

db_top_dir = "C:\\Users\\Administrator\\Desktop\\ku\\1-2\\XAI604_2025_spring\\"
test_top_dir = os.path.join(db_top_dir, "stop_music\\music_test0")

test_dataset = sample_util.make_dataset(test_top_dir)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=
    "C:\\Users\\Administrator\\Desktop\\ku\\1-2\\XAI604_2025_spring\\project\\models\\checkpoint-5000"
)

for data in test_dataset:
    ref = data["labels"]
    hyp = transcriber(data["input_values"])
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
