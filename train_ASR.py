import torch
import whisper
from whisper.utils import get_writer
from utils.dataset import ASR_input
import os
import re
from difflib import SequenceMatcher
import numpy as np
import librosa
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import argparse

"""
python train_ASR.py --stage prepare --max_samples 100

python train_ASR.py --stage train --start_epoch 0 --num_epochs 1
python train_ASR.py --stage train --start_epoch 1 --num_epochs 1
python train_ASR.py --stage train --start_epoch 2 --num_epochs 1
python train_ASR.py --stage train --start_epoch 3 --num_epochs 1
python train_ASR.py --stage train --start_epoch 4 --num_epochs 1
python train_ASR.py --stage train --start_epoch 5 --num_epochs 1
python train_ASR.py --stage train --start_epoch 6 --num_epochs 1
python train_ASR.py --stage train --start_epoch 7 --num_epochs 1
python train_ASR.py --stage train --start_epoch 8 --num_epochs 1
python train_ASR.py --stage train --start_epoch 9 --num_epochs 1

python train_ASR.py --stage evaluate --max_samples 20
"""

def load_training_data(answer_file, max_samples=None):
    """
    Useage:
        return a list of ASR_input object
    Args:
        answer_file: Path to the answer file
        max_samples: Maximum number of samples to load (if None, load all)
    Returns:
        List of ASR_input objects (checkout /utils/dataset.py/ASR_input)
    """
    data = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                id_text = line.strip().split('\t')
                if len(id_text) == 2:
                    try:
                        audio_id = int(id_text[0])
                        sentence = id_text[1]
                        sample = ASR_input(audio_id, sentence)
                        if sample is not None:
                            data.append(sample)
                    except Exception as e:
                        print(f"Error processing line {i+1}: {str(e)}")
                        continue
    return data

def tokenize_text(text):
    # Split English words and keep Chinese characters as individual tokens
    words = re.findall(r'[A-Za-z]+|[\u4e00-\u9fff]|[0-9]+', text)
    return words

def calculate_ser(reference, hypothesis):
    # Tokenize both texts
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)
    
    # Calculate S, D, I using SequenceMatcher
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens)
    operations = matcher.get_opcodes()
    
    S = D = I = 0
    for tag, i1, i2, j1, j2 in operations:
        if tag == 'replace':
            S += max(i2-i1, j2-j1)  # Count substitutions
        elif tag == 'delete':
            D += i2-i1  # Count deletions
        elif tag == 'insert':
            I += j2-j1  # Count insertions
    
    N = len(ref_tokens)  # Total reference tokens
    
    # Calculate SER
    ser = (S + D + I) / N if N > 0 else 1.0
    
    return {
        'substitutions': S,
        'deletions': D,
        'insertions': I,
        'total_ref_tokens': N,
        'ser': ser
    }

def evaluate_model(model, dataset):
    predictions = {}
    references = {}
    timestamps_all = {}
    ser_scores = {}
    total_ser = 0
    
    for sample in dataset:
        filename = sample["ans_id"]
        audio_array = sample['audio']['array']
        
        # Transcribe audio
        result = model.transcribe(audio_array, word_timestamps=True)
        
        pred_text = result["text"].strip()
        ref_text = sample["sentence"].strip()
        
        # Calculate SER
        ser_result = calculate_ser(ref_text, pred_text)
        ser_scores[filename] = ser_result
        total_ser += ser_result['ser']
        
        # Store results
        predictions[filename] = pred_text
        references[filename] = ref_text
        timestamps_all[filename] = result["segments"]
    
    avg_ser = total_ser / len(dataset) if dataset else 0
    return ser_scores, avg_ser, predictions, references, timestamps_all

def apply_audio_augmentation(audio_array, sample_rate):
    # Apply audio augmentation techniques
    augmented = audio_array.copy()
    
    # Randomly choose augmentation
    aug_type = random.choice(['noise', 'speed', 'pitch', 'none'])
    
    if aug_type == 'noise':
        # Add random noise
        noise_level = random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_level, len(augmented))
        augmented = augmented + noise
    
    elif aug_type == 'speed':
        # Speed perturbation
        speed_factor = random.uniform(0.9, 1.1)
        augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
    
    elif aug_type == 'pitch':
        # Pitch shift
        n_steps = random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=n_steps)
    
    return augmented

def prepare_training(answer_file, max_samples=100, output_dir="models/whisper_finetuned"):
    """
    Prepare training data and model
    Returns:
        model: Whisper model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save models
    """
    print("Loading base model...")
    model_path = os.path.join("models", "medium.pt")
    if os.path.exists(model_path):
        print(f"Using local model file: {model_path}")
        model = whisper.load_model("medium", download_root="models", in_memory=False)
    else:
        print("Local model file not found. Please download the model file manually:")
        print("https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
        print(f"And place it in: {model_path}")
        raise FileNotFoundError("Model file not found")
    
    print("Loading training data...")
    dataset = load_training_data(answer_file, max_samples=max_samples)
    print(f"Created dataset with {len(dataset)} samples")
    
    # Split dataset
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    return model, train_dataset, val_dataset, output_dir

def train_epochs(model, train_dataset, val_dataset, output_dir, 
                start_epoch=0, num_epochs=10, learning_rate=5e-6,
                eval_interval=10, patience=3, weight_decay=0.01, dropout_rate=0.1):
    """
    Train model for specified number of epochs
    """
    # Set up training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Enable dropout during training
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    
    # Prepare optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Load checkpoint if exists
    if start_epoch > 0:
        checkpoint_path = os.path.join(output_dir, f"whisper_epoch_{start_epoch}.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from epoch {start_epoch}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Training loop
    best_val_ser = float('inf')
    best_model_path = None
    patience_counter = 0
    
    # Whisper parameters
    SAMPLE_RATE = 16000
    max_sample_length = 30 * SAMPLE_RATE  # 30 seconds at 16kHz
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0
        processed_batches = 0
        
        for batch_idx, sample in enumerate(train_dataset):
            try:
                # Get audio array and ensure it's the right shape
                audio_array = sample['audio']['array']
                if not isinstance(audio_array, np.ndarray):
                    print(f"Warning: audio_array is not numpy array, type: {type(audio_array)}")
                    continue
                
                # Print audio shape for debugging
                print(f"Original audio shape: {audio_array.shape}")
                
                # Resample if necessary
                if sample['audio']['sampling_rate'] != SAMPLE_RATE:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample['audio']['sampling_rate'],
                        target_sr=SAMPLE_RATE
                    )
                
                # Convert to tensor and ensure correct shape
                audio_tensor = torch.from_numpy(audio_array).float()
                if len(audio_tensor.shape) > 1:
                    audio_tensor = audio_tensor.mean(dim=-1)  # Convert stereo to mono if needed
                
                # Pad or trim the audio tensor
                if audio_tensor.shape[0] > max_sample_length:
                    audio_tensor = audio_tensor[:max_sample_length]
                elif audio_tensor.shape[0] < max_sample_length:
                    padding = max_sample_length - audio_tensor.shape[0]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                
                print(f"Processed audio shape: {audio_tensor.shape}")
                
                # Convert to mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_tensor).to(device)
                print(f"Mel spectrogram shape: {mel.shape}")
                
                # Get target text
                target_text = sample["sentence"]
                
                # Encode target text
                tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
                target_tokens = tokenizer.encode(target_text)
                target_tokens = torch.tensor(target_tokens).unsqueeze(0).to(device)
                
                # Forward pass
                logits = model(mel.unsqueeze(0), target_tokens)
                
                # Calculate loss
                decoder_input = target_tokens[:, :-1]
                target_tokens_shifted = target_tokens[:, 1:]
                logits = logits[:, :decoder_input.shape[1], :]
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_tokens_shifted.reshape(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                processed_batches += 1
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Evaluate periodically
                if (batch_idx + 1) % eval_interval == 0:
                    print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                print(f"Audio array type: {type(audio_array)}, shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'no shape'}")
                continue
        
        if processed_batches > 0:
            # Calculate average training loss
            avg_train_loss = total_loss / processed_batches
            print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
        else:
            print("No batches were successfully processed in this epoch")
            continue
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            try:
                val_ser_scores, avg_val_ser, _, _, _ = evaluate_model(model, val_dataset)
                print(f"Validation SER: {avg_val_ser:.4f}")
                
                # Update learning rate
                scheduler.step(avg_val_ser)
                
                # Save if better
                if avg_val_ser < best_val_ser:
                    best_val_ser = avg_val_ser
                    patience_counter = 0
                    best_model_path = os.path.join(output_dir, f"whisper_best_ser_{avg_val_ser:.4f}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_train_loss,
                        'val_ser': avg_val_ser,
                    }, best_model_path)
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
            except Exception as e:
                print(f"Error during validation: {str(e)}")
                continue
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"whisper_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_train_loss,
            'val_ser': avg_val_ser if 'avg_val_ser' in locals() else float('inf'),
        }, checkpoint_path)
    
    return model, best_model_path

def evaluate_model_performance(model, answer_file, max_samples=20):
    """
    Evaluate model performance on test set
    """
    print("\nRunning evaluation...")
    test_dataset = load_training_data(answer_file, max_samples=max_samples)
    
    ser_scores, avg_ser, predictions, references, timestamps_all = evaluate_model(model, test_dataset)
    
    # Print results (first 8 files)
    print("\nEvaluation Results:")
    for i, filename in enumerate(ser_scores):
        if i >= 8:
            break
        print(f"\nFile {filename}:")
        print(f"Substitutions: {ser_scores[filename]['substitutions']}")
        print(f"Deletions: {ser_scores[filename]['deletions']}")
        print(f"Insertions: {ser_scores[filename]['insertions']}")
        print(f"Total Reference Tokens: {ser_scores[filename]['total_ref_tokens']}")
        print(f"SER: {ser_scores[filename]['ser']:.4f}")
        print("\nPrediction:")
        for segment in timestamps_all[filename]:
            print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
        print("\nReference:")
        print(references[filename])
    
    print(f"\nAverage SER across all test files: {avg_ser:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Whisper ASR Training')
    parser.add_argument('--stage', type=str, required=True, 
                      choices=['prepare', 'train', 'evaluate'],
                      help='Training stage to execute')
    parser.add_argument('--start_epoch', type=int, default=0,
                      help='Starting epoch for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--max_samples', type=int, default=100,
                      help='Maximum number of samples to use')
    parser.add_argument('--output_dir', type=str, default='models/whisper_finetuned',
                      help='Directory to save models')
    args = parser.parse_args()
    
    answer_file = "data/TRAINING_DATASET_1_PHASE/Training_Dataset_01/task1_answer.txt"
    
    if args.stage == 'prepare':
        model, train_dataset, val_dataset, output_dir = prepare_training(
            answer_file, 
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
        # Save datasets for later use
        torch.save({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }, os.path.join(output_dir, 'datasets.pt'))
        
    elif args.stage == 'train':
        # Load model and datasets
        model = whisper.load_model("medium", device="cpu", download_root="models")
        # weights_only=False is important for loading the datasets on windows
        datasets = torch.load(os.path.join(args.output_dir, 'datasets.pt'), weights_only=False)
        train_dataset = datasets['train_dataset']
        val_dataset = datasets['val_dataset']
        
        # Train for specified epochs
        model, best_model_path = train_epochs(
            model, train_dataset, val_dataset, args.output_dir,
            start_epoch=args.start_epoch,
            num_epochs=args.num_epochs
        )
        
    elif args.stage == 'evaluate':
        # Load best model
        model = whisper.load_model("medium", device="cpu", download_root="models")
        best_model_path = os.path.join(args.output_dir, "whisper_best_ser_*.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        evaluate_model_performance(model, answer_file, max_samples=args.max_samples)

if __name__ == "__main__":
    main() 