import json
import torch
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from collections import defaultdict

# Constants
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
BOS_TOKEN = '<|endoftext|>'
EOS_TOKEN = '<|END|>'
PAD_TOKEN = '<|pad|>'
SEP_TOKEN = '\n\n####\n\n'

# PHI categories for validation
TRAIN_PHI_CATEGORY = [
    'PATIENT', 'DOCTOR', 'USERNAME', 'FAMILYNAME', 'PROFESSION',
    'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY',
    'DISTRICT', 'COUNTY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
    'AGE', 'DATE', 'TIME', 'DURATION', 'SET',
    'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
    'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
    'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM',
    'OTHER'
]

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
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
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval().cuda()
    else:
        print("CUDA not available, running on CPU")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            trust_remote_code=True
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    
    return model, tokenizer

def get_anno_format(infos, audio_timestamps):
    """Convert model output to annotation format with timestamps"""
    anno_list = []
    phi_dict = defaultdict(list)
    
    for line in infos.split("\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in TRAIN_PHI_CATEGORY and value:
            phi_dict[key].append(value)

    remaining_timestamps = audio_timestamps.copy()
    used_indices = set()

    for phi_key, phi_values in phi_dict.items():
        for phi_value in phi_values:
            phi_tokens = phi_value.lower().strip().split()

            for i in range(len(remaining_timestamps) - len(phi_tokens) + 1):
                if any((i + j) in used_indices for j in range(len(phi_tokens))):
                    continue

                match = True
                for j, phi_token in enumerate(phi_tokens):
                    tsd_word = remaining_timestamps[i + j]['word'].replace("Ġ", "").replace("▁", "").strip().lower()
                    if tsd_word != phi_token:
                        match = False
                        break

                if match:
                    anno_list.append({
                        "phi": phi_key,
                        "st_time": remaining_timestamps[i]['start'],
                        "ed_time": remaining_timestamps[i + len(phi_tokens) - 1]['end'],
                        "entity": phi_value
                    })
                    for j in range(len(phi_tokens)):
                        used_indices.add(i + j)
                    break
    return anno_list

def predict(model, tokenizer, input_data, audio_timestamps, template="<|endoftext|> __CONTENT__\n\n####\n\n"):
    """Generate predictions for input data"""
    seeds = [template.replace("__CONTENT__", data['content']) for data in input_data]
    sep = tokenizer.sep_token
    eos = tokenizer.eos_token
    pad = tokenizer.pad_token
    model.eval()
    device = model.device
    
    texts = tokenizer(seeds, return_tensors='pt', padding=True).to(device)
    outputs = []

    with torch.no_grad():
        output_tokens = model.generate(
            **texts,
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
        
        for idx, pred in enumerate(preds):
            if "Null" in pred:
                continue
            phi_infos = pred[pred.index(sep)+len(sep):].replace(pad, "").replace(eos, "").strip()
            annotations = get_anno_format(phi_infos, audio_timestamps[input_data[idx]['fid']]['segments'])
            for annotation in annotations:
                outputs.append(
                    f'{input_data[idx]["fid"]}\t{annotation["phi"]}\t{annotation["st_time"]}\t{annotation["ed_time"]}\t{annotation["entity"]}'
                )
    return outputs

def main():
    # Load model and tokenizer
    model_path = "models/ner_model/best_adapter"  # or the path to your saved model
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load validation data
    valid_data = load_dataset(
        "csv",
        data_files="data/task1_answer.txt",
        delimiter='\t',
        features=Features({
            'fid': Value('string'),
            'content': Value('string')
        }),
        column_names=['fid', 'content']
    )
    valid_list = list(valid_data['train'])

    # Load audio timestamps
    with open('data/task1_answer_timestamps.json', 'r', encoding='utf-8') as file:
        audio_timestamps = json.load(file)

    # Generate predictions
    BATCH_SIZE = 10
    with open("data/task2_answer.txt", 'w', encoding='utf8') as f:
        for i in tqdm(range(0, len(valid_list), BATCH_SIZE)):
            with torch.no_grad():
                data = valid_list[i:i+BATCH_SIZE]
                outputs = predict(model, tokenizer, data, audio_timestamps)
                for o in outputs:
                    f.write(o)
                    f.write('\n')

if __name__ == "__main__":
    main()
