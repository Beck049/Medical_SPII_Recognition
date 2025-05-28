import os
import json
from .audio import get_wav_file
"""
This file is used to modify the data format
- ASR_input
- NER_tsv
"""

def ASR_input(id, sentence):
  """
  Dict Layout:
  {
    "audio": {
      "array": audio_array,
      "file_id": file_id,
      "sampling_rate": sampling_rate,
      "duration": duration
    },
    "ans_id": id,
    "sentence": sentence
  }
  """
  audio = get_wav_file(f"{id}.wav", 0)['audio']
  return {
    "audio": audio,
    "ans_id": id,
    "sentence": sentence
  }

def NER_tsv(task2_answer_file, task1_answer_file, output_file, max_samples=None):
    """
    Convert NER annotation data to TSV format for training.
    
    Args:
        task2_answer_file (str): Path to task2_answer.txt containing NER annotations
        task1_answer_file (str): Path to task1_answer.txt containing transcriptions
        output_file (str): Path to save the TSV file
        max_samples (int, optional): Maximum number of samples to process
    
    The task2_answer.txt format is:
    file_id    PHI_TYPE    start_time    end_time    entity_text
    
    The task1_answer.txt format is:
    file_id    transcription_text
    
    The output TSV format will be:
    fid    content    label
    where label is formatted as: "PHI_TYPE:entity_text\nPHI_TYPE:entity_text"
    """
    # Load transcriptions
    transcriptions = {}
    with open(task1_answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                fid, text = line.strip().split('\t', 1)
                transcriptions[fid] = text
    
    # Process NER annotations
    current_fid = None
    current_annotations = []
    samples = []
    sample_count = 0
    
    with open(task2_answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples is not None and sample_count >= max_samples:
                break
                
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 5:  # Valid NER annotation line
                    fid, phi_type, start_time, end_time, entity = parts
                    
                    if current_fid != fid:
                        # Save previous sample if exists
                        if current_fid is not None and current_fid in transcriptions:
                            label = '\\n'.join(current_annotations)
                            samples.append({
                                'fid': current_fid,
                                'content': transcriptions[current_fid],
                                'label': label
                            })
                            sample_count += 1
                        
                        # Start new sample
                        current_fid = fid
                        current_annotations = []
                    
                    # Add annotation
                    if phi_type != "NULL":  # Skip NULL annotations
                        current_annotations.append(f"{phi_type}:{entity}")
    
    # Save last sample if exists
    if current_fid is not None and current_fid in transcriptions:
        label = '\\n'.join(current_annotations)
        samples.append({
            'fid': current_fid,
            'content': transcriptions[current_fid],
            'label': label
        })
    
    # Write to TSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(f"{sample['fid']}\t{sample['content']}\t{sample['label']}\n")
    
    print(f"Generated {len(samples)} samples in {output_file}")
    return samples

def load_ner_data(tsv_file, max_samples=None):
    """
    Load NER training data from TSV file.
    
    Args:
        tsv_file (str): Path to the TSV file
        max_samples (int, optional): Maximum number of samples to load
    
    Returns:
        list: List of dictionaries containing 'fid', 'content', and 'label'
    """
    samples = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                fid, content, label = line.strip().split('\t')
                samples.append({
                    'fid': fid,
                    'content': content,
                    'label': label
                })
    return samples

# test
if __name__ == "__main__":
    print(ASR_input(19, "Any overture of something that's kind of like a little white flag or peace offering to just get a week of peace, I'm not talking about permanent I'm going to placate and cow tow to you and to talk my needs in other... No. Just talking about lets..."))