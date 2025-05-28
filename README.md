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

### NER finetune input
```python
{
  "10": [
    {'phi': "ID_NUMBER",
     'start_time': "2.533",
     'end_time': "3.154",
     'entity': "09F016547J"},
    {'phi': "ID_NUMBER",
     'start_time': "7.632",
     'end_time': "8.124",
     'entity': "09F016547J"},
  ],
  "13": [
    {'phi': "NULL"}
  ],
}
```
|SHI 類別|類別定義|競賽提供的資料中的類別名稱|
|---|---|---|
|姓名|病患名 / 醫師名 / 使用者名稱 / 家屬姓名 / 個人姓名|PATIENT / DOCTOR / USERNAME / FAMILYNAME / PERSONALNAME|
|職業|無|PROFESSION|
|地點|診間號 / 部門 / 醫院 / 組織 / 街 / 城市 / 區 / 郡 / 州 / 國家 / 區號 / 其他|ROOM / DEPARTMENT / HOSPITAL / ORGANIZATION / STREET / CITY / DISTRICT / COUNTY / STATE / COUNTRY / ZIP / LOCATION-OTHER|
|年齡|無|AGE|
|日期|日期 / 時間 / 週期 / 頻率|DATE / TIME / DURATION / SET|
|聯絡方式|手機號碼 / 傳真 / 電子郵件信箱 / 網址 / 網際網路協定位址|PHONE / FAX / EMAIL / URL / IPADDRESS|
|識別符|社群安全碼 / 醫療紀錄號碼 / 健康計畫號碼 / 帳戶 / 證照號碼 / 車牌 / 裝置號碼 / 生物識別碼 / 識別碼|SOCIAL_SECURITY_NUMBER / MEDICAL_RECORD_NUMBER / HEALTH_PLAN_NUMBER / ACCOUNT_NUMBER / LICENSE_NUMBER / VEHICLE_ID / DEVICE_ID / BIOMETRIC_ID / ID_NUMBER|

### NER output
|File ID|SHI Type|Start Offset|End Offset|Text (Optional)|
|---|---|---|---|---|
|file01|AGE|1.523|1.826|57|
|file01|PATIENT|1.923|2.120|Ken Moll|
|file01|IDNUM|4.85|5.401|62S021442H|
|file01|STREET|5.52|5.865|Yale|
|file01|CITY|6.451|6.732|Andergrove|
|file01|STATE|6.735|6.957|Tasmania|
|file01|ZIP|11.167|11.984|2042|
|file01|MEDICALRECORD|12.491|13.644|6270214.MFH|
|file01|IDNUM|14.464|15.782|62S02144|