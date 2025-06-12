# CHiME-5 기반 WebDataset 음성 인식 및 발화 의도 분석용 데이터셋

본 데이터셋은 CHiME-5 코퍼스를 기반으로 구성된 WebDataset 형식의 음성 인식 및 발화 의도 분석용 데이터셋입니다.  
음성 발화 단위를 `start_time` 및 `end_time` 기준으로 segment하여 `.wav` 형식으로 저장하고, 그에 대응되는 전사 정보를 `.txt`로 함께 제공합니다.

---

## 📌 데이터셋 채택 이유

- **CHiME-5**는 실제 가정 환경에서 녹음된 다인 대화 음성을 포함하며, 현실적인 노이즈 환경에서의 자동 음성 인식(ASR) 및 음성 이해(SLU)에 적합한 대표적인 벤치마크입니다.
- 발화별 시간 정보가 상세하게 주어져 있어, **세밀한 단위의 발화 인식 및 태깅 학습에 적합**합니다.
- 또한, 실제 사용자 간의 자연스러운 대화 흐름, 겹말(overlap), 비유창성(disfluency)이 존재하기 때문에 **컨텍스트 기반의 복합 모델 학습**에 효과적입니다.


---

## 🔍 샘플 구성

각 `.tar` 파일(shard)은 WebDataset 포맷을 따르며, 하나의 샘플은 다음과 같이 구성됩니다:

| Key | 설명 |
|-----|------|
| `__key__` | 유니크 샘플 식별자 (UUID) |
| `wav` | `.wav` 형식의 발화 오디오 (byte stream) |
| `txt` | 대응되는 텍스트 전사 (str) |

Python 예시:

```python
import webdataset as wds

dataset = (
    wds.WebDataset("wds/train/shard-000000.tar")
    .decode()
    .to_tuple("wav", "txt")
)

for audio, text in dataset:
    print(text)  # 전사 출력
```

## 🧪 학습 및 추론 사용 방식

| 용도 | 사용 split |
|------|------------|
| 모델 학습 (`trainer.train()`) | ✅ `train/` + `dev/` |
| 성능 검증 (`eval_dataset`, WER 추적 등) | ✅ `dev/` |
| 모델 추론 (예: test-time decoding) | ✅ `eval/` |

- `train` 및 `dev` split은 모두 전사 라벨이 포함되어 있으며, 학습 및 검증 단계에 사용됩니다.
- `dev`는 `eval_dataset`으로 할당되어 모델의 성능(WER 등)을 평가하고, `load_best_model_at_end` 기준이 됩니다.
- `eval` split은 학습에 사용되지 않으며, **최종 디코딩 결과 평가 또는 예측 테스트**에 사용됩니다.

---

## ⚙️ 전처리 및 생성 정보

- **원본 데이터셋**: [CHiME-5 공식 사이트](https://www.chimechallenge.org/datasets/chime5)
- **오디오 segment 기준**: `start_time` 및 `end_time` (JSON 기반)
- **사용 도구**:
  - `pydub` (오디오 segment 추출)
  - `webdataset` (샤드 포맷 저장)
  - `transformers`, `evaluate` (Hugging Face 기반 학습)
- **변환 스크립트**: `create_chime5_webdataset.py`

---