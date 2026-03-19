import os
import shutil
import easyocr
from tqdm import tqdm

SOURCE_DIR = "./No_duplicates"
REJECT_DIR = "./Rejected_watermarks"
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# gpu = True for hardware acceleration, False if it crashes
reader = easyocr.Reader(['en'], gpu=True)

def setup_reject_directories():
    if not os.path.exists(REJECT_DIR):
        os.makedirs(REJECT_DIR)
    for emotion in TARGET_EMOTIONS:
        path = os.path.join(REJECT_DIR, emotion)
        if not os.path.exists(path):
            os.makedirs(path)

def main():
    setup_reject_directories()
    images_to_check = []
    for emotion in TARGET_EMOTIONS:
        emotion_dir = os.path.join(SOURCE_DIR, emotion)
        if os.path.exists(emotion_dir):
            for filename in os.listdir(emotion_dir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    images_to_check.append((os.path.join(emotion_dir, filename), emotion, filename))
                    
    total_images = len(images_to_check)
    if total_images == 0:
        print("No images found")
        return
        
    print(f"\nScanning {total_images} images")
    removed_count = 0
    for img_path, emotion, filename in tqdm(images_to_check, desc="OCR Scanning"):
        try:
            # detail = 0 returns a simple list of detected text strings
            results = reader.readtext(img_path, detail=0)
            if len(results) > 0:
                detected_text = "".join(results).strip()
                # reject threshold (discard image if detected text > 2 characters)
                if len(detected_text) > 2:
                    dst_path = os.path.join(REJECT_DIR, emotion, filename)
                    shutil.move(img_path, dst_path)
                    removed_count += 1      
        except Exception as e:
            print(f"Could not scan {filename} due to error: {e}")
            continue

    print(f"Scanned images: {total_images}")
    print(f"Watermarks detected and removed: {removed_count}")
    print(f"Clean dataset remains in: {SOURCE_DIR}")
    print(f"Rejected files moved to: {REJECT_DIR}")

if __name__ == "__main__":
    main()