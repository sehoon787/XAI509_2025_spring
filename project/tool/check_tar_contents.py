import tarfile
from pathlib import Path
import io
import torchaudio

# TAR_DIR = Path("C:/Users/Administrator/Desktop/ku/1-2/XAI509_2025_spring/data/train")
TAR_DIR = Path("D:/ku/1-2/XAI509_2025_spring/data/train")
tar_files = sorted(TAR_DIR.glob("shard-*.tar"))

if not tar_files:
    print("❌ .tar 파일이 존재하지 않습니다.")
else:
    for tar_path in tar_files:
        print(f"\n📦 {tar_path.name} 검사 시작:")
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                for member in sorted(members, key=lambda m: m.name):
                    if not member.name.endswith(".wav"):
                        continue

                    try:
                        wav_bytes = tar.extractfile(member).read()
                        torchaudio.load(io.BytesIO(wav_bytes))
                        print(f"  ✅ {member.name} (정상)")
                    except Exception as e:
                        print(f"  ❌ {member.name} (깨짐): {e}")
        except Exception as e:
            print(f"  ⚠️ {tar_path.name} 열기 실패: {e}")
