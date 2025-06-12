import tarfile
from pathlib import Path

def decompress(file_path, save_path: str):
    tar_gz = Path(file_path)
    shard_dir = tar_gz.parent / save_path
    if not shard_dir.exists():
        shard_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_gz, "r:gz") as tar:
            tar.extractall(path=shard_dir)
        print(f"Extracted shards to: {shard_dir}")

if __name__ == '__main__':
    decompress(file_path="./CHiME5/CHiME5_train.tar.gz", save_path="train")
    decompress(file_path="./CHiME5/CHiME5_dev.tar.gz", save_path="dev")
    decompress(file_path="./CHiME5/CHiME5_eval.tar.gz", save_path="eval")
    decompress(file_path="./CHiME5/CHiME5_transcriptions.tar.gz", save_path="transcriptions")
