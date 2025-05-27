import whisper
import librosa
from utils.dataset import ASR_input
from utils.audio import get_valid_audio_name_sorted
import argparse
import os
import torch
import glob

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
        Transcribed text
    """
    # Load and resample audio
    audio_array, sr = librosa.load(audio_path, sr=16000)
    
    # Transcribe
    result = model.transcribe(audio_array, word_timestamps=True, verbose=False)
    return result["text"].strip()

def main():
    parser = argparse.ArgumentParser(description='Run ASR on validation dataset')
    parser.add_argument('--use_finetuned', action='store_true',
                      help='Use finetuned model instead of pretrained')
    args = parser.parse_args()

    # Load model
    model = load_model(args.use_finetuned)

    # Get validation audio files
    validation_dir = "data/TRAINING_DATASET_1_PHASE/Validation_Dataset/audio"
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")
    
    audio_files = get_valid_audio_name_sorted()
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {validation_dir}")

    # Create output directory if it doesn't exist
    output_dir = "outputs/ASR"
    os.makedirs(output_dir, exist_ok=True)

    # Process each audio file
    print("\nProcessing validation audio files...")
    output_file = os.path.join(output_dir, "validation_predictions.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for audio_file in audio_files:
            audio_id = os.path.splitext(audio_file)[0]  # Get filename without extension
            audio_path = os.path.join(validation_dir, audio_file)
            
            print(f"Processing {audio_file}...")
            try:
                # Transcribe audio
                transcribed_text = transcribe_audio(model, audio_path)
                
                # Write to file in format: id\ttext
                f.write(f"{audio_id}\t{transcribed_text}\n")
                print(f"Transcribed: {transcribed_text[:100]}...")  # Print first 100 chars
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

    print(f"\nTranscription complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
