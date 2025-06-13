import json
import uuid
from pathlib import Path
from io import BytesIO
import soundfile as sf
import webdataset as wds

# 사용자 지정 경로
DATA_TYPE = "train"     # train, dev, eval
DATA_ROOT = Path("D:/ku/1-2/XAI509_2025_spring/data/CHiME5")
AUDIO_ROOT = DATA_ROOT / "train" / "CHiME5" / "audio" / DATA_TYPE
TRANS_ROOT = DATA_ROOT / "transcriptions" / "CHiME5" / "transcriptions" / DATA_TYPE
OUTPUT_ROOT = Path(f"D:/ku/1-2/XAI509_2025_spring/data/{DATA_TYPE}")

# 마이크 설정
MIC_ID = "U04"
CHANNEL = "CH3"  # 마이크 채널

# 시간 변환 함수
def time_to_sec(t: str) -> float:
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# 오디오 슬라이스 함수
def extract_audio_segment(wav_path, start_time, end_time):
    with sf.SoundFile(str(wav_path), 'r') as f:
        sr = f.samplerate
        start_frame = int(start_time * sr)
        end_frame = int(end_time * sr)
        f.seek(start_frame)
        frames = f.read(end_frame - start_frame)
        buf = BytesIO()
        sf.write(buf, frames, sr, format='WAV')
        return buf.getvalue()

# 샘플 생성
def generate_samples():
    for json_path in sorted(TRANS_ROOT.glob("S*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            utterances = json.load(f)

        session_id = json_path.stem  # S17
        mic_wav = f"{session_id}_{MIC_ID}.{CHANNEL}.wav"
        wav_path = AUDIO_ROOT / mic_wav

        if not wav_path.exists():
            print(f"[WARN] No wav found: {wav_path}")
            continue

        for utt in utterances:
            try:
                start = time_to_sec(utt["start_time"][MIC_ID])
                end = time_to_sec(utt["end_time"][MIC_ID])
                text = utt.get("words", "").strip()
                if not text:
                    continue
                audio_bytes = extract_audio_segment(wav_path, start, end)

                uid = f"{session_id}_{MIC_ID}_{str(uuid.uuid4())[:8]}"
                yield {
                    "__key__": uid,
                    "wav": audio_bytes,
                    "txt": text
                }
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

# WebDataset으로 저장
def write_webdataset():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shard_path = "file:" + OUTPUT_ROOT.as_posix() + "/shard-%06d.tar"
    with wds.ShardWriter(shard_path, maxcount=1000) as sink:
        for sample in generate_samples():
            sink.write(sample)

if __name__ == "__main__":
    write_webdataset()
