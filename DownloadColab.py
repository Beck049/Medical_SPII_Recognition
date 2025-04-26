import gdown

files_info = [
    {
        "file_id": "1Vc4rtbMxM_DQzIi1LATrpj4Tn-t5L_DG",
        "file_name": "AI_CUP_ASR_Whisper.ipynb"
    },
    {
        "file_id": "17SSKvr0nsuRo8yabdg8RH-BSPNMmVBnL",
        "file_name": "AI_CUP_NER_Deepseek.ipynb"
    }
]

for file in files_info:
    download_url = f"https://drive.google.com/uc?id={file['file_id']}"
    gdown.download(download_url, file['file_name'], quiet=False)
