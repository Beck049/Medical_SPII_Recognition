# 將訓練資料格式轉換成 stanza NER fine-tune 所需格式（BIO）

import os
import jieba
from collections import defaultdict


def load_features(path):
    features = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            file_id, text = line.strip().split('\t', 1)
            features[file_id] = text
    return features

def load_labels(path):
    labels = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                file_id, shi_type, _, _, entity_text = parts[:5]
                labels[file_id].append((entity_text, shi_type))
    return labels

def convert_to_bio(text, entities):
    words = list(jieba.cut(text))
    labels = ['O'] * len(words)

    for entity_text, entity_type in entities:
        entity_words = list(jieba.cut(entity_text))
        for i in range(len(words)):
            if words[i:i + len(entity_words)] == entity_words:
                labels[i] = f'B-{entity_type}'
                for j in range(1, len(entity_words)):
                    labels[i + j] = f'I-{entity_type}'
                break 
    return list(zip(words, labels))

def write_bio_output(features, labels, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        for file_id, text in features.items():
            ents = labels.get(file_id, [])
            bio_pairs = convert_to_bio(text, ents)
            for word, tag in bio_pairs:
                out.write(f"{word} {tag}\n")
            out.write("\n") 


feature_file = './data/TRAINGING DATASET_1PHASE/Training_Dataset_01/task1_answer.txt'
label_file = './data/TRAINGING DATASET_1PHASE/Training_Dataset_01/task2_answer.txt'
output_file = '../outputs/NER/train_data.bio'

features = load_features(feature_file)
labels = load_labels(label_file)
write_bio_output(features, labels, output_file)

print(f"BIO 格式資料已輸出至 {output_file}")
