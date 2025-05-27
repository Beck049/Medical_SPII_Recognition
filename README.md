# Medical_SPII_Recognition
AI Cup 2025 - 醫病語音敏感個人資料辨識競賽

## Hierarchy
```python
Medical_SPII_Recognition/
    ├- colabTest/ # 搬運完後刪除
    |
    ├- config/
    |     ├─ config_ASR.yaml  # 放 model 路徑、參數設定
    |     └─ config_NER.yaml  # 放 model 路徑、參數設定
    |
    ├- data/   # data 不要上傳到 github
    |    └- TRAINING_DATASET_1_PHASE/
    |               ├- Training_Dataset_01/
    |               |        ├─ audio/
    |               |        |     └── xxx.wav
    |               |        ├- task1_answer.txt
    |               |        └─ task2_answer.txt
    |               ├- Training_Dataset_02/
    |               └──Validation_Dataset/
    |                        └─ audio/
    |                             └── xxx.wav
    ├- models/
    |     └── whisper_finetuned/
    |
    ├- outputs/
    |     └── ASR/
    |
    ├- utils/       # some common function
    |
    ├- .gitignore
    ├- config.yaml  # 全域設定使用到套件
    ├- init.sh      # 下載必要套件
    |
    ├- run_ASR.py   # 階段 1: 音檔轉文字
    ├- run_NER.py   # 階段 2: 文字抽取敏感資料
    ├- train_ASR.py   # ASR finetune
    ├- train_NER.py   # NER finetune
    |
    └- README.md
```

## JSON

### ASR finetune input
```python
{
  "audio": {
    "array": [0.803150659427047e-05,
          0.0002169541548937559,
          0.0003004224272444844,
          0.00028273859061300755,
          0.00016738526755943894,
          0.0343711841851473e-05, ...
        ],
    "file_id": 19,
    "sampling_rate": 48000,
    "duration": 16.331
  },
  "ans_id": 19,
  "sentence": "Any overture of something that's kind of like a little white flag or peace offering to just get a week of peace, I'm not talking about permanent "I'm going to placate and cow tow to you and to talk my needs in other..." No. Just talking about lets..."
}
```
### ASR output 1
```python
{id} '\t' {sentence} '\n' 
```

## Workflow
1. Run `DownloadColab.py` to sync the Python notebook
2. Check if the change is correct, then `git push`
