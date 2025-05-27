# XLM-RoBERTa finetune 

# requirement
# pip install transformers datasets seqeval accelerate


from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, load_dataset, DatasetDict
import numpy as np
from seqeval.metrics import classification_report, f1_score
import stanza

def tokenize_and_align_labels(example):
    '''
    處理 subword 對齊標籤，因為 XLM-R 是基於 subword 的 tokenizer
    '''
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label_to_id[example["ner_tags"][word_idx]])
        else:
            labels.append(-100)  # subword
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions)
    }

def predict(sentence):
    tokens = list(sentence)
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    outputs = model(**inputs).logits
    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()
    
    result = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label = id_to_label[predictions[idx]]
        if label != "O":
            result.append((tokens[word_idx], label))
    return result

# path
train_data_path = "./outputs/NER/train_data_20.json"
save_model_path = "./outputs/NER/XLM-R_finetuned"

# 載入模型與 tokenizer
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# load data
dataset = load_dataset("json", data_files=train_data_path, field=None, split="train")
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# label 與整數轉換對應
label_list = list(sorted({
    label
    for split in dataset.values()   
    for ex in split                 
    for label in ex["ner_tags"]     
}))
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# Tokenize 資料（處理 subword）
tokenized_dataset = dataset.map(tokenize_and_align_labels)

# 建 model 跟 Trainer
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id
)

# 目前還沒調參
training_args = TrainingArguments(
    output_dir="./xlmr-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# train
trainer.train()

# 儲存 model
trainer.save_model(save_model_path)      
tokenizer.save_pretrained(save_model_path) 

# # 預測測試 1 筆資料
# # 用 stanza.MultilingualPipeline() 做句子斷詞
# nlp = stanza.MultilingualPipeline()
# test_sen = "Yeah, I imagine it would — sorry, go ahead. So it's supposed to work immediately, right? Yep. So we'll see if I'm productive tomorrow. I hope I'm productive today. I've actually been trying to plan. If I do the titles today, then I can do my laundry tomorrow. Right. I probably could bring my computer and do titles while I'm doing my laundry. If I was — but I won't do that."
# # test_sen = "It makes it really hard when I lose things. Even this morning, he knocked over my glasses. I had them on the dresser and he was trying to find them this morning because I can't see anything without my glasses. He was like, "Are you sure you didn't put them somewhere else?" and I'm like, "Yes, I put them in one of two places. They are in neither. I heard you knock them over last night when you came back and the lights were off." He got annoyed that I was blaming him and I got annoyed that he got annoyed for me blaming him when it was his fault. He eventually found them obviously because I'm wearing them but that was a stressful way to wake up."
# doc = nlp(test_sen)
# test_sen_tokens = [word.text for sent in doc.sentences for word in sent.words]
# print(predict(test_sen_tokens))

