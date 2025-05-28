import whisper
import librosa
from utils.dataset import ASR_input
from utils.audio import get_valid_audio_name_sorted
import argparse
import os
import torch
import glob
import json

def load_model(use_finetuned=False):
    """
    Load Whisper model, either pretrained or finetuned
    Args:
        use_finetuned: Whether to use finetuned model
    Returns:
        Loaded Whisper model
    """
    if use_finetuned:
        # Find the best model checkpoint
        model_dir = "models/whisper_finetuned"
        model_files = glob.glob(os.path.join(model_dir, "whisper_best_ser_*.pt"))
        if not model_files:
            raise FileNotFoundError("No finetuned model found. Please train first.")
        # Sort by SER (lower is better) and get the best one
        model_path = sorted(model_files, key=lambda x: float(x.split('_')[-1].replace('.pt', '')))[0]
        print(f"Loading finetuned model from {model_path}")
        # Load base model first
        model = whisper.load_model("medium")
        # Load finetuned weights
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Loading pretrained medium model")
        model = whisper.load_model("medium")
    
    return model

def transcribe_audio(model, audio_path):
    """
    Transcribe a single audio file
    Args:
        model: Whisper model
        audio_path: Path to audio file
    Returns:
        dict containing transcribed text and timestamps
    """
    # Load and resample audio
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    # Transcribe with word timestamps
    result = model.transcribe(audio_array, word_timestamps=True, verbose=False)
    
    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        for word in segment["words"]:
            segments.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
    
    return {
        "text": result["text"].strip(),
        "segments": segments
    }

def main():
    parser = argparse.ArgumentParser(description='Run ASR on validation dataset')
    parser.add_argument('--use_finetuned', action='store_true',
                      help='Use finetuned model instead of pretrained')
    parser.add_argument('--data_dir', type=str, default='data/TRAINING_DATASET_1_PHASE/Training_Dataset_02',
                      help='Directory containing audio files')
    args = parser.parse_args()

    # Load model
    model = load_model(args.use_finetuned)

    # Get audio files
    audio_dir = os.path.join(args.data_dir, "audio")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    audio_files = get_valid_audio_name_sorted()
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    # Create output directory if it doesn't exist
    output_dir = "outputs/ASR"
    os.makedirs(output_dir, exist_ok=True)

    # Process each audio file
    print("\nProcessing audio files...")
    output_file = os.path.join(output_dir, "predictions.txt")
    timestamps_file = os.path.join(args.data_dir, "task1_answer_timestamps.json")
    
    # Dictionary to store timestamps for each file
    timestamps_dict = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for audio_file in audio_files:
            audio_id = os.path.splitext(audio_file)[0]  # Get filename without extension
            audio_path = os.path.join(audio_dir, audio_file)
            
            print(f"Processing {audio_file}...")
            try:
                # Transcribe audio with timestamps
                result = transcribe_audio(model, audio_path)
                transcribed_text = result["text"]
                
                # Store timestamps
                timestamps_dict[audio_id] = {
                    "segments": result["segments"]
                }
                
                # Write to file in format: id\ttext
                f.write(f"{audio_id}\t{transcribed_text}\n")
                print(f"Transcribed: {transcribed_text[:100]}...")  # Print first 100 chars
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

    # Save timestamps to JSON file
    with open(timestamps_file, 'w', encoding='utf-8') as f:
        json.dump(timestamps_dict, f, ensure_ascii=False, indent=2)

    print(f"\nTranscription complete. Results saved to {output_file}")
    print(f"Timestamps saved to {timestamps_file}")

if __name__ == "__main__":
    main()
