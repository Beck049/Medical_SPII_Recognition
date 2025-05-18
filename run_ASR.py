import whisper
import librosa
from jiwer import mer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from utils.dataset import ASR_input

# Create test dataset
test_dataset = [
    ASR_input(19, "Any overture of something that's kind of like a little white flag or peace offering to just get a week of peace, I'm not talking about permanent I'm going to placate and cow tow to you and to talk my needs in other... No. Just talking about lets...")
]

# 載入 Whisper 官方模型 (你可以改成 'tiny', 'small', 'medium', 'large')
model = whisper.load_model("small")

def resample_audio(audio_array, original_sampling_rate):
    if original_sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=original_sampling_rate, target_sr=16000)
    return audio_array

def calculate_mer(ground_truth_texts, predicted_texts):
    mer_scores = {}
    total_mer = 0
    count = 0

    normalizer = BasicTextNormalizer()

    for filename, ref_text in ground_truth_texts.items():
        if filename in predicted_texts:
            pred_text = predicted_texts[filename]
            ref_text = normalizer(ref_text)
            pred_text = normalizer(pred_text)
            mer_score = mer(ref_text, pred_text)
            mer_scores[filename] = mer_score
            total_mer += mer_score
        else:
            mer_scores[filename] = 1
            total_mer += 1
        count += 1

    average_mer = total_mer / count if count != 0 else 0
    return mer_scores, average_mer

def evaluate_mer_with_timestamps(model, dataset):
    predictions = {}
    references = {}
    timestamps_all = {}

    for sample in dataset:
        filename = sample["ans_id"]
        audio_array = resample_audio(sample['audio']['array'], sample['audio']['sampling_rate'])

        # need numpy array
        result = model.transcribe(audio_array, word_timestamps=False, verbose=False)

        predictions[filename] = result["text"].strip()
        references[filename] = sample["sentence"].strip()
        timestamps_all[filename] = result["segments"]

    mer_scores, avg_mer = calculate_mer(references, predictions)
    return mer_scores, avg_mer, predictions, references, timestamps_all

mer_scores, avg_mer, predictions, references, timestamps_all = evaluate_mer_with_timestamps(model, test_dataset)

# 輸出結果
print("All MER scores:\n", mer_scores)
print("Average MER score:\n", avg_mer)

for filename in predictions:
    print(f"\nFilename: {filename}")
    print("Prediction:")
    for segment in timestamps_all[filename]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
    print("Reference:")
    print(references[filename])
