import os
from .audio import get_wav_file

def ASR_input(id, sentence):
  audio = get_wav_file(f"{id}.wav", 0)['audio']
  return {
    "audio": audio,
    "ans_id": id,
    "sentence": sentence
  }

# test
if __name__ == "__main__":
    print(ASR_input(19, "Any overture of something that's kind of like a little white flag or peace offering to just get a week of peace, I'm not talking about permanent I'm going to placate and cow tow to you and to talk my needs in other... No. Just talking about lets..."))