import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    AutoConfig,
    get_scheduler
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from collections import Counter
from utils.dataset import NER_tsv, load_ner_data
from run_NER import load_model_and_tokenizer, predict

"""
python train_NER.py --stage prepare --max_samples 100

python train_NER.py --stage train --start_epoch 0 --num_epochs 1
python train_NER.py --stage train --start_epoch 1 --num_epochs 1
python train_NER.py --stage train --start_epoch 2 --num_epochs 1
python train_NER.py --stage train --start_epoch 3 --num_epochs 1
python train_NER.py --stage train --start_epoch 4 --num_epochs 1
python train_NER.py --stage train --start_epoch 5 --num_epochs 1
python train_NER.py --stage train --start_epoch 6 --num_epochs 1
python train_NER.py --stage train --start_epoch 7 --num_epochs 1
python train_NER.py --stage train --start_epoch 8 --num_epochs 1
python train_NER.py --stage train --start_epoch 9 --num_epochs 1    

python train_NER.py --stage evaluate --max_samples 20
"""

# Constants
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
BOS_TOKEN = '<|endoftext|>'
EOS_TOKEN = '<|END|>'
PAD_TOKEN = '<|pad|>'
SEP_TOKEN = '\n\n####\n\n'

# Training hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-4
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 6

def parse_args():
    parser = argparse.ArgumentParser(description='Train NER model with staged execution')
    parser.add_argument('--stage', type=str, required=True, choices=['prepare', 'train', 'evaluate'],
                      help='Stage to execute: prepare, train, or evaluate')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to use (for prepare and evaluate stages)')
    parser.add_argument('--start_epoch', type=int, default=0,
                      help='Starting epoch for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                      help='Number of epochs to train')
    parser.add_argument('--model_path', type=str, default='models/ner_model',
                      help='Path to save/load model')
    parser.add_argument('--data_dir', type=str, default='data/TRAINING_DATASET_1_PHASE/Training_Dataset_02',
                      help='Directory containing task1_answer.txt and task2_answer.txt')
    return parser.parse_args()

def setup_model_and_tokenizer():
    """Initialize model and tokenizer with special tokens"""
    special_tokens_dict = {
        'eos_token': EOS_TOKEN,
        'bos_token': BOS_TOKEN,
        'pad_token': PAD_TOKEN,
        'sep_token': SEP_TOKEN
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        output_hidden_states=False
    )

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            config=config,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            trust_remote_code=True
        )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer

def collate_batch_with_prompt_template(batch, tokenizer, template="<|endoftext|> __CONTENT__\n\n####\n\n__LABEL__ <|END|>", IGNORED_PAD_IDX=-100):
    """Collate function for dataloader"""
    texts = [template.replace("__LABEL__", data['label']).replace("__CONTENT__", data['content']) 
             for data in list(batch)]
    encoded_seq = tokenizer(texts, padding=True)

    indexed_tks = torch.tensor(encoded_seq['input_ids'])
    attention_mask = torch.tensor(encoded_seq['attention_mask'])
    encoded_label = torch.tensor(encoded_seq['input_ids'])
    encoded_label[encoded_label == tokenizer.pad_token_id] = IGNORED_PAD_IDX

    return indexed_tks, encoded_label, attention_mask

def prepare_data(args):
    """Prepare and analyze training data"""
    print("Preparing training data...")
    
    # Define file paths
    task2_answer_file = os.path.join(args.data_dir, "task2_answer.txt")
    task1_answer_file = os.path.join(args.data_dir, "task1_answer.txt")
    output_file = "data/task2_train.tsv"
    
    # Convert data to TSV format
    samples = NER_tsv(
        task2_answer_file=task2_answer_file,
        task1_answer_file=task1_answer_file,
        output_file=output_file,
        max_samples=args.max_samples
    )
    
    # Print data statistics
    ctr = Counter()
    for sample in samples:
        phi_labelwvalue = sample['label'].split("\\n")
        phi_label = [j.split(":")[0] for j in phi_labelwvalue]
        ctr.update(phi_label)
    print("\nClass distribution:", ctr)
    
    # Save data statistics
    stats = {
        'num_samples': len(samples),
        'class_distribution': dict(ctr)
    }
    os.makedirs('data/stats', exist_ok=True)
    with open('data/stats/ner_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nData statistics saved to data/stats/ner_stats.json")
    return samples

def train_model(model, tokenizer, train_data, output_dir, start_epoch, num_epochs):
    """Main training loop with specified epochs"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch_with_prompt_template(batch, tokenizer),
        pin_memory=True
    )

    # Load checkpoint if exists
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path) and start_epoch > 0:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        early_stop_counter = checkpoint['early_stop_counter']
        print(f"Loaded checkpoint from epoch {start_epoch-1}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        best_loss = float("inf")
        early_stop_counter = 0

    total_steps = len(train_dataloader) * num_epochs // GRADIENT_ACCUMULATION_STEPS
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=total_steps,
    )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0
        model.train()

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for step, (seqs, labels, masks) in progress_bar:
            input_ids = seqs.to(device)
            labels = labels.to(device)
            attention_mask = masks.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

            progress_bar.set_postfix(
                loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,
                lr=lr_scheduler.get_last_lr()[0]
            )

        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'early_stop_counter': early_stop_counter
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_path = os.path.join(output_dir, "best_adapter")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"Best model saved at {best_path} with loss {avg_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop patience: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def evaluate_model(model, tokenizer, max_samples=None, data_dir=None):
    """Evaluate model on validation data"""
    print("Evaluating model...")
    
    # Load validation data
    task1_answer_file = os.path.join(data_dir, "task1_answer.txt")
    valid_data = load_dataset(
        "csv",
        data_files=task1_answer_file,
        delimiter='\t',
        features=Features({
            'fid': Value('string'),
            'content': Value('string')
        }),
        column_names=['fid', 'content']
    )
    
    if max_samples:
        valid_data['train'] = valid_data['train'].select(range(min(max_samples, len(valid_data['train']))))
        print(f"Using {len(valid_data['train'])} samples for evaluation")

    # Load audio timestamps
    timestamps_file = os.path.join(data_dir, "task1_answer_timestamps.json")
    with open(timestamps_file, 'r', encoding='utf-8') as file:
        audio_timestamps = json.load(file)

    # Generate predictions
    BATCH_SIZE = 10
    predictions = []
    valid_list = list(valid_data['train'])
    
    for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
        with torch.no_grad():
            data = valid_list[i:i+BATCH_SIZE]
            outputs = predict(model, tokenizer, data, audio_timestamps)
            predictions.extend(outputs)

    # Save predictions
    os.makedirs('data/predictions', exist_ok=True)
    with open('data/predictions/ner_predictions.txt', 'w', encoding='utf8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    print(f"Saved {len(predictions)} predictions to data/predictions/ner_predictions.txt")

def main():
    args = parse_args()
    
    if args.stage == 'prepare':
        prepare_data(args)
    
    elif args.stage == 'train':
        # Load dataset using the generated TSV file
        train_data = load_dataset(
            "csv",
            data_files="data/task2_train.tsv",
            delimiter='\t',
            features=Features({
                'fid': Value('string'),
                'content': Value('string'),
                'label': Value('string')
            }),
            column_names=['fid', 'content', 'label']
        )
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Train model
        train_model(model, tokenizer, train_data['train'], args.model_path, args.start_epoch, args.num_epochs)
    
    elif args.stage == 'evaluate':
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(os.path.join(args.model_path, 'best_adapter'))
        # Pass data_dir to evaluate_model
        evaluate_model(model, tokenizer, args.max_samples, data_dir=args.data_dir)

if __name__ == "__main__":
    main()
