import os
import shutil
from PIL import Image
from transformers import pipeline
from tqdm import tqdm
import torch

INPUT_DIR = "./No_duplicates"
OUTPUT_CLEAN_DIR = "./No_children"
OUTPUT_REJECTED_DIR = "./Rejected_children"
TARGET_CLASSES_TO_REJECT = ["0-2", "3-9"]

device = 0 if torch.cuda.is_available() else -1 # setup hardware acceleration
age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier", device=device)
valid_extensions = ('.png', '.jpg', '.jpeg')

def main():
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                files_to_process.append(os.path.join(root, file))

    total_files = len(files_to_process)
    if total_files == 0:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {total_files} images")
    kept_count = 0
    rejected_count = 0

    for input_path in tqdm(files_to_process, desc="Filtering dataset"):
        try:
            rel_path = os.path.relpath(input_path, INPUT_DIR)
            image = Image.open(input_path).convert("RGB")
            results = age_classifier(image)
            # extract highest probability age group
            top_label = results[0]['label']
            if top_label in TARGET_CLASSES_TO_REJECT:
                output_path = os.path.join(OUTPUT_REJECTED_DIR, rel_path)
                rejected_count += 1
            else:
                output_path = os.path.join(OUTPUT_CLEAN_DIR, rel_path)
                kept_count += 1
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)    
        except Exception as e:
            print(f"Failed to classify {rel_path}: {e}")
            continue

    print(f"Kept (adults): {kept_count}")
    print(f"Rejected (children): {rejected_count} ")
    print(f"Clean dataset saved in:: {OUTPUT_CLEAN_DIR}")
    print(f"Rejected files moved to: {OUTPUT_REJECTED_DIR}")

if __name__ == "__main__":
    main()