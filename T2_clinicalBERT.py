from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import AutoModelForTokenClassification

# 測試中，未完成

# 載入 tokenizer 與模型
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

ner_model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

text = "Patient John Smith was diagnosed with pneumonia on January 5th."
results = ner_pipeline(text)

for ent in results:
    print(f"{ent['word']} → {ent['entity_group']} ({ent['score']:.2f})")