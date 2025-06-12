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
    decompress(file_path=r"C:\Users\Administrator\Desktop\ku\1-2\XAI604_2025_spring\stop_music\music_test0.tar.gz",
               save_path="music_test0")
    decompress(file_path=r"C:\Users\Administrator\Desktop\ku\1-2\XAI604_2025_spring\stop_music\music_train.tar.gz",
               save_path="music_train")
