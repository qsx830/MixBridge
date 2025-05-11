import os
import torch
import clip
import numpy as np
from PIL import Image
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B-32.pt", device=device)
model = torch.nn.DataParallel(model)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).to(device)
        return image_input, image_path

def get_image_embedding(image_input):
    with torch.no_grad():
        embedding = model.module.encode_image(image_input)
    return embedding / embedding.norm(dim=-1, keepdim=True)

def calculate_mse(embedding1, embedding2):
    return mse_loss(embedding1, embedding2)

def calculate_clip_score(embedding1, embedding2):
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

def process_images(folder_path, external_images, batch_size=32):
    all_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    if len(all_images) == 0:
        return 0,0
    success = 0

    clip_scores = []

    dataset = ImageDataset(all_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    external_embeddings = [get_image_embedding(preprocess_image(img)) for img in external_images]

    for image_batch, image_paths in dataloader:

        image_embeddings = get_image_embedding(image_batch)

        for i, image_embedding in enumerate(image_embeddings):
            best_mse = float('inf')
            best_match_embedding = None

            for external_embedding in external_embeddings:
                mse = calculate_mse(image_embedding, external_embedding)
                if mse < best_mse:
                    best_mse = mse
                    best_match_embedding = external_embedding

            if best_match_embedding is not None:
                clip_score = calculate_clip_score(image_embedding, best_match_embedding)
                clip_scores.append(clip_score)
                if clip_score > 0.7:
                    success += 1

            # print(f"Processed {image_paths[i]}")

    average_clip_score = np.mean(clip_scores)
    asr = success / len(all_images)
    return average_clip_score, asr

def process_single_folder(folder_path, external_images, batch_size=32):
    print(f"Processing subfolder: {folder_path}")
    avg_clip_score, asr = process_images(folder_path+'/recon', external_images, batch_size)
    print("Score:",avg_clip_score)
    print("ASR:",asr)
    # f.write(f"{subfolder}: {avg_clip_score:.4f}, ASR: {asr:.4f}\n")
    # print(f"Saved {subfolder} average CLIP score")

def main():
    import sys

    if len(sys.argv) != 2:
        print("Missing Parameters")
        sys.exit(1)

    folder_path = sys.argv[1]

    external_images = ['backdoor_image1.png', 'backdoor_image2.png', 'backdoor_image3.png']

    process_single_folder(folder_path, external_images, batch_size=128)

if __name__ == "__main__":
    main()
