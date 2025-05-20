import stanza
import re
import csv

# 多語言處理，目前僅有斷詞
text = "今天 today Morrison pouch有水懷疑LiverRuptureNormalsaline先加500好家明醫師車禍女性二十歲右膝laceration三乘三然後左膝二乘二擦傷然後沒有其他擦傷緹娜是有個lacerationcase幫她suture一下好我來我們推到A4小姐有沒有怎麼樣哪裡不舒服"
# text = "Yeah, I imagine it would — sorry, go ahead. So it's supposed to work immediately, right? Yep. So we'll see if I'm productive tomorrow. I hope I'm productive today. I've actually been trying to plan. If I do the titles today, then I can do my laundry tomorrow. Right. I probably could bring my computer and do titles while I'm doing my laundry. If I was — but I won't do that."

# 下載模型
stanza.download(lang="multilingual")

# MultilingualPipeline()
nlp = stanza.MultilingualPipeline()

doc = nlp(text)
print("偵測語言:", doc.lang)
for sent in doc.sentences:
    print("分詞:", [word.text for word in sent.words])
    for ent in sent.ents:
        print(f"NER: {ent.text} -> {ent.type}")

# 中英分段（錯誤）
# # 下載模型
# stanza.download('en')  # 英文
# stanza.download('zh')  # 中文
# # Regex 中英切段
# def split_zh_en(text):
#     pattern = r'([\u4e00-\u9fff，。！？、]+|[a-zA-Z0-9\s\'\-\.]+)'
#     matches = re.findall(pattern, text)
#     return [seg.strip() for seg in matches if seg.strip()]

# # 偵測語言
# def detect_lang(segment):
#     if re.search(r'[\u4e00-\u9fff]', segment):
#         return 'zh'
#     else:
#         return 'en'

# # BIO 標註工具
# def bio_tagging(sentence_words, entities):
#     labels = ['O'] * len(sentence_words)
#     for ent in entities:
#         if ent.type not in allowed_entity_types:
#             continue  # 忽略非指定類別
#         start_char, end_char, ent_type = ent.start_char, ent.end_char, ent.type
#         for idx, word in enumerate(sentence_words):
#             if word.start_char >= start_char and word.end_char <= end_char:
#                 if word.start_char == start_char:
#                     labels[idx] = f'B-{ent_type.lower()}'
#                 else:
#                     labels[idx] = f'I-{ent_type.lower()}'
#     return labels

# # 讀取輸入檔並處理
# input_file = '/Users/antingchao/Documents/GitHub/Medical_SPII_Recognition/data/TRAINGING DATASET_1PHASE/Training_Dataset_01/task1_answer.txt'
# output_file = './stanza_output.tsv'

# # 欲保留的實體類別（stanza 預設使用大寫）
# # 目前僅保留與目標相關的類別
# allowed_entity_types = {'PERSON', 'FAC', 'ORG', 'LOC', 'GPE', 'DATE', 'TIME'}

# # 下載模型
# stanza.download('en')  # 英文
# stanza.download('zh')  # 中文

# # 初始化Pipeline，包含 tokenize, pos, ner
# nlp_en = stanza.Pipeline(lang='en', processors='tokenize,pos,ner')
# nlp_zh = stanza.Pipeline(lang='zh', processors='tokenize,pos,ner')


# with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8', newline='') as fout:
#     tsv_writer = csv.writer(fout, delimiter='\t')
#     tsv_writer.writerow(['sentence_id', 'token', 'label'])

#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             sentence_id, sentence = line.split('\t', 1)
#         except ValueError:
#             print(f"跳過格式錯誤行: {line}")
#             continue

#         segments = split_zh_en(sentence)

#         for seg in segments:
#             lang = detect_lang(seg)
#             nlp = nlp_zh if lang == 'zh' else nlp_en
#             doc = nlp(seg)

#             for sent in doc.sentences:
#                 words = sent.words
#                 tokens = [w.text for w in words]
#                 labels = bio_tagging(words, sent.ents if sent.ents else [])
#                 for word, label in zip(tokens, labels):
#                     tsv_writer.writerow([sentence_id, word, label])
