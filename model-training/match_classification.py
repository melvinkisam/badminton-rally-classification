import os
import cv2
import imageio
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from model_package import load_model
from tqdm import tqdm
from model_package import VideoChunkDataset

# Paths
output_dir = 'badminton-rally-classification/model-training/output'
model_path = 'badminton-rally-classification/model-training/trained_models/trained_model_3.pth'
match_path = 'badminton-rally-classification/dataset-creation/match-videos/match_9-sb.mp4'

class_names = ['interval', 'rally', 'shuttlechange', 'floormopping', 'setbreak']
 

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(model_path, map_location=device)

model = load_model(num_classes=5)  # Must match architecture used before
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
loss = checkpoint['loss']

model.eval()
model.to(device)


dataset = VideoChunkDataset(match_path, chunk_frames=128, sample_frames=32)
loader = DataLoader(dataset, batch_size=4, shuffle=False)


num_classes = len(class_names)
frame_ranges = {i: [] for i in range(num_classes)} # Dictionary of lists to store classified frame ranges

with torch.no_grad():
    for clips, starts in tqdm(loader, desc="Classifying..."):
        outputs = model(clips)
        preds = torch.argmax(outputs, dim=1)

        for pred, start_idx in zip(preds.cpu(), starts):
            start = start_idx.item()
            end = start + dataset.chunk_frames
            frame_ranges[pred.item()].append((start, end))


cap = cv2.VideoCapture(match_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for class_idx, ranges in frame_ranges.items():
    out_path = os.path.join(output_dir, f"{class_names[class_idx]}.mp4")
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height)) # Use mp4v compression (opt1)
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264')  # Use libx264 compression

    for start, end in ranges:
        for f in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
            # writer.write(frame) # opt1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV) to RGB (imageio expects RGB) opt2
            writer.append_data(frame_rgb) # opt2

    # writer.release() # opt1
    writer.close() # opt2
    print(f"✅ Saved class {class_idx} video to: {out_path}")

cap.release()