import os
import sys
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file with progress bar
    """
    try:
        print(f"Connecting to {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            print("Warning: Could not determine file size")
            total_size = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Downloading to {filename}...")
        
        with open(filename, 'wb') as f, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                pbar.update(size)
        
        print(f"Download complete! File saved to {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"Error saving file: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    # Whisper medium model URL
    model_url = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
    model_path = os.path.join("models", "medium.pt")
    
    print("Starting Whisper model download...")
    if download_file(model_url, model_path):
        print("Model download successful!")
    else:
        print("Model download failed!", file=sys.stderr)
        sys.exit(1) 