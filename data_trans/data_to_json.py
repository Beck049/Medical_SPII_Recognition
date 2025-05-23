# fine-tune XLM-roberta 用
# 將訓練資料轉成 json，格式：
'''
[
  {
    "tokens": ["I", "'m", "going", "to", "work", "tomorrow"],
    "ner_tags": ["O", "O", "O", "O", "O", "B-DATE"]
  },
  ...
]
'''

import json
import stanza
from collections import defaultdict
from tqdm import tqdm

# data path
feature_file = '../data/TRAINGING DATASET_1PHASE/Training_Dataset_01/task1_answer_20.txt'
label_file = '../data/TRAINGING DATASET_1PHASE/Training_Dataset_01/task2_answer_20.txt'
output_file = '../outputs/NER/train_data.json'

# 這行（下載模型）執行過一次就好，下次可以註解掉，不用重新下載
# stanza.download('multilingual')

nlp = stanza.MultilingualPipeline()

# load data
transcripts = {}
with open(feature_file, "r", encoding="utf-8") as f:
    for line in f:
        file_id, text = line.strip().split("\t")
        transcripts[file_id] = text

ner_dict = defaultdict(list)
with open(label_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            continue
        file_id, ent_type, _, _, ent_text = parts
        ner_dict[file_id].append((ent_text, ent_type))

ner_data = []
for file_id, text in tqdm(transcripts.items()):
    doc = nlp(text)
    tokens = [word.text for sent in doc.sentences for word in sent.words]

    token_used = [False] * len(tokens)
    labels = ["O"] * len(tokens)

    for ent_text, ent_type in ner_dict[file_id]:
        ent_doc = nlp(ent_text)
        ent_tokens = [word.text for sent in ent_doc.sentences for word in sent.words]
        ent_len = len(ent_tokens)

        found = False
        for i in range(len(tokens) - ent_len + 1):
            if tokens[i:i+ent_len] == ent_tokens and not any(token_used[i:i+ent_len]):
                labels[i] = f"B-{ent_type}"
                for j in range(1, ent_len):
                    labels[i + j] = f"I-{ent_type}"
                for j in range(ent_len):
                    token_used[i + j] = True
                found = True
                break
        if not found:
            print(f"Warning: entity '{ent_text}' not found in file {file_id}")

    ner_data.append({
        "tokens": tokens,
        "ner_tags": labels
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ner_data, f, ensure_ascii=False, indent=2)

print(f"json 檔已輸出至 {output_file}")
