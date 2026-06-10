import os
import shutil
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SOURCE_DIR = "./Raw_dataset"
TARGET_DIR = "./No_duplicates"
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def setup_target_directories():
    """
    Creates the main target folder and subfolders for each emotion
    if they do not exist already.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)
    for emotion in TARGET_EMOTIONS:
        os.makedirs(os.path.join(TARGET_DIR, emotion), exist_ok=True)

def compute_hash(image_path, emotion):
    """
    Calculates a unique visual fingerprint (hash) for a single image.
    This helps in identifying images that look exactly the same.
    """
    # perceptual hash (pHash) to find visual duplicates
    try:
        with Image.open(image_path) as img:
            img_hash = str(imagehash.phash(img))
        return image_path, emotion, img_hash, None
    except Exception as e:
        return image_path, emotion, None, str(e)

def process_and_deduplicate():
    """
    Finds all images in the source folders, calculates their hashes using
    multiple threads for speed, and copies only the unique images to the target folder.
    """
    print("Starting deduplication")
    setup_target_directories()
    # collection of all valid image paths from the source directories
    images_to_process = []
    for emotion in TARGET_EMOTIONS:
        emotion_dir = os.path.join(SOURCE_DIR, emotion)
        if not os.path.exists(emotion_dir):
            continue
        for filename in os.listdir(emotion_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_to_process.append((os.path.join(emotion_dir, filename), emotion))
                
    total_images = len(images_to_process)
    if total_images == 0:
        print("No images found")
        return
        
    print(f"Found {total_images} images")
    stats = {emo: {'total': 0, 'kept': 0, 'removed': 0, 'errors': 0} for emo in TARGET_EMOTIONS}
    seen_hashes = set()
    
    # thread pool for speeding up the hashing process
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(compute_hash, path, emo) for path, emo in images_to_process]
        for future in tqdm(as_completed(futures), total=total_images, desc="Processing images"):
            path, emotion, img_hash, error = future.result()
            stats[emotion]['total'] += 1
            if error:
                stats[emotion]['errors'] += 1
                continue
                
            # core deduplication logic (keeps the image only if its visual hash is new)
            if img_hash in seen_hashes:
                stats[emotion]['removed'] += 1
            else:
                seen_hashes.add(img_hash)
                stats[emotion]['kept'] += 1
                filename = os.path.basename(path)
                target_path = os.path.join(TARGET_DIR, emotion, filename)
                # prevent SameFileError
                if os.path.abspath(path) != os.path.abspath(target_path):
                    shutil.copy2(path, target_path)
                    
    total_kept = 0
    total_removed = 0
    
    for emo in TARGET_EMOTIONS:
        total_kept += stats[emo]['kept']
        total_removed += stats[emo]['removed']
    
    print(f"Seen: {total_images} Kept: {total_kept} Duplicates: {total_removed}")
    print(f"Clean dataset is saved to: {TARGET_DIR}")

if __name__ == '__main__':
    process_and_deduplicate()