import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# get a list of sorted wav files
def get_audio_name_sorted():
    folder_path = os.path.join(PROJECT_ROOT, 'data', 'TRAINING_DATASET_1_PHASE', 'Training_Dataset_01', 'audio')
    
    #  get all .wav
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    def extract_number(filename):
        return int(filename.split('.')[0])
    
    # sort
    wav_files_sorted = sorted(wav_files, key=extract_number)
    return wav_files_sorted

# draw waveform
def draw_wavform(wav_file_name, audio_array, sr):
    print(f'File name: {wav_file_name}')

    lt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio_array, sr=sr)
    plt.title("Waveform")
    plt.show()

    print(f'Mean amplitude: {np.mean(np.abs(audio_array))}\n')

    num_zeros = np.sum(audio_array == 0)
    print(f'Zero rate: {round(num_zeros * 100 / len(audio_array), 3)}% ( {num_zeros}/{len(audio_array)} )')
    return None

# transform wav into json
def get_wav_file(filename, show=False):
    audio_dir = os.path.join(PROJECT_ROOT, 'data', 'TRAINING_DATASET_1_PHASE', 'Training_Dataset_01', 'audio')
    wav_file_path = os.path.join(audio_dir, filename)

    if not os.path.isfile(wav_file_path):
        print(f"File {wav_file_path} not found.")
        return None

    audio_array, sr = librosa.load(wav_file_path, sr=None)

    if show:
        draw_wavform(filename, audio_array, sr)

    return {
        "audio": {
            "array": audio_array,
            "file_id": int(os.path.splitext(filename)[0]),
            "sampling_rate": sr,
            "duration": librosa.get_duration(y=audio_array, sr=sr)
        }
    }

# test
if __name__ == "__main__":
    files = get_audio_name_sorted()
    print(f'file count: {len(files)}')
    print(files[:10])
    print(get_wav_file(files[0]))