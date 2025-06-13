from transformers import pipeline, AutoProcessor
import evaluate
import os
import sample_util

# 경로 설정
db_top_dir = "C:/Users/Administrator/Desktop/ku/1-2/XAI509_2025_spring"
# db_top_dir = "D:/ku/1-2/XAI509_2025_spring/"
test_top_dir = os.path.join(db_top_dir, "data/dev")

# processor 불러오기
processor = AutoProcessor.from_pretrained(
    os.path.join(db_top_dir, "project/models/checkpoint-10000")
)

# 데이터셋 로드
test_dataset = sample_util.make_dataset(test_top_dir)

# 평가 지표 로드
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ASR 파이프라인
transcriber = pipeline(
    "automatic-speech-recognition",
    model=os.path.join(db_top_dir, "project/models/checkpoint-10000"),
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor
)

# 결과 저장
refs = []
hyps = []

# 각 샘플 처리
for data in test_dataset:
    label_ids = data["labels"]
    input_values = data["input_values"]

    # 음성 인식 결과
    hyp = transcriber(input_values)["text"]

    # 레이블 디코딩
    label_ids = [i if i != -100 else processor.tokenizer.pad_token_id for i in label_ids]
    ref = processor.decode(label_ids, skip_special_tokens=True)

    refs.append(ref)
    hyps.append(hyp)

    # 개별 WER, CER 출력
    sample_wer = wer_metric.compute(predictions=[hyp], references=[ref])
    sample_cer = cer_metric.compute(predictions=[hyp], references=[ref])
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
    print(f"WER: {sample_wer:.4f} | CER: {sample_cer:.4f}")
    print("-" * 50)

# 전체 평균 WER, CER 출력
total_wer = wer_metric.compute(predictions=hyps, references=refs)
total_cer = cer_metric.compute(predictions=hyps, references=refs)

print(f"\n### Final Average WER: {total_wer:.4f} ###")
print(f"### Final Average CER: {total_cer:.4f} ###")
