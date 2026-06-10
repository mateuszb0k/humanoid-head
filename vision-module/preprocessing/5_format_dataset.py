import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SOURCE_DIR = "./No_cartoons"
TARGET_DIR = "./Final_dataset"
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
TARGET_SIZE = (96, 96)

def setup_target_directories():
    """
    Creates the main target folder and subfolders for each emotion
    where the final resized images will be saved.
    """
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    for emotion in TARGET_EMOTIONS:
        emotion_path = os.path.join(TARGET_DIR, emotion)
        if not os.path.exists(emotion_path):
            os.makedirs(emotion_path)

def process_single_image(src_path, dst_path):
    """
    Reads a single image, converts it to grayscale, resizes it to the target size,
    and saves it to the new destination.
    """
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Corrupted file: {src_path}"
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst_path, resized_img)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """
    Finds all valid images and processes them in parallel using multiple threads
    to quickly format the entire dataset.
    """
    print(f"Target format: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px, Grayscale\n")
    setup_target_directories()
    # map source and target paths for all valid images
    tasks = []
    for emotion in TARGET_EMOTIONS:
        emotion_dir = os.path.join(SOURCE_DIR, emotion)
        if os.path.exists(emotion_dir):
            for filename in os.listdir(emotion_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(emotion_dir, filename)
                    new_filename = os.path.splitext(filename)[0] + ".jpg"
                    dst_path = os.path.join(TARGET_DIR, emotion, new_filename)
                    tasks.append((src_path, dst_path))

    total_images = len(tasks)
    if total_images == 0:
        print("No images found in source directory")
        return

    print(f"Found {total_images} images to process")
    success_count = 0
    error_count = 0
    
    # launching on multiple threads to make it faster
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_single_image, src, dst): (src, dst) for src, dst in tasks}
        for future in tqdm(as_completed(futures), total=total_images, desc="Processing", unit="img"):
            success, error_msg = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1

    print(f"Successfully processed: {success_count}")
    print(f"Skipped (errors): {error_count}")
    print(f"Final dataset output: {TARGET_DIR}")

if __name__ == "__main__":
    main()