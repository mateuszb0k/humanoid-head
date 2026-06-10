import os
import shutil
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

INPUT_DIR = "./No_children"
OUTPUT_CLEAN_DIR = "./No_cartoons"
OUTPUT_REJECTED_DIR = "./Rejected_cartoons"
THRESHOLD = 0.8 # minimum confidence required to classify an image as a drawing (0-1)

device = "cuda" if torch.cuda.is_available() else "cpu" # setup hardware acceleration
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
valid_extensions = ('.png', '.jpg', '.jpeg')

def main():
    """
    Analyzes each image to calculate how much it looks like a drawing.
    If the score is too high, the image is moved to the rejected folder.
    """
    files_to_process = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                files_to_process.append(os.path.join(root, file))

    total_files = len(files_to_process)
    if total_files == 0:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {total_files} images to scan")
    kept_count = 0
    rejected_count = 0
    # define search classes for CLIP
    labels = ["a photo of a real human face", "a drawing, cartoon, anime, illustration or sketch"]

    for input_path in tqdm(files_to_process, desc="Analyzing style"):
        try:
            rel_path = os.path.relpath(input_path, INPUT_DIR)
            image = Image.open(input_path).convert("RGB")
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            drawing_score = probs[0, 1].item()
            if drawing_score > THRESHOLD:
                output_path = os.path.join(OUTPUT_REJECTED_DIR, rel_path)
                rejected_count += 1
            else:
                output_path = os.path.join(OUTPUT_CLEAN_DIR, rel_path)
                kept_count += 1
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(input_path, output_path)    
        except Exception as e:
            print(f"Skipping {rel_path} due to error: {e}")
            continue

    print(f"Kept (real photos): {kept_count}")
    print(f"Rejected (cartoons/animations): {rejected_count}")
    print(f"Clean dataset saved in:: {OUTPUT_CLEAN_DIR}")
    print(f"Rejected files moved to: {OUTPUT_REJECTED_DIR}")

if __name__ == "__main__":
    main()