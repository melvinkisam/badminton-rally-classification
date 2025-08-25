import os
import cv2
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_package import load_model, MatchChunkDataset

# Paths
output_dir = '/cs/home/psxmk12/badminton_classification/output_pth416'
model_path = '/cs/home/psxmk12/badminton_classification/trained_models/trained_model_4.pth'
match_path = '/cs/home/psxmk12/badminton_classification/match_9-sb.mp4'

class_names = ['interval', 'rally', 'shuttlechange', 'floormopping', 'setbreak']

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

model = load_model(num_classes=5)  # Must match architecture used before
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1
loss = checkpoint['loss']

model.eval()
model.to(device)

# Dataset / DataLoader
dataset = MatchChunkDataset(match_path, chunk_frames=16, sample_frames=16)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

num_classes = len(class_names)
frame_ranges = {i: [] for i in range(num_classes)}

# Run classification
with torch.no_grad():
    for clips, starts in tqdm(loader, desc="Classifying..."):
        clips = clips.to(device)
        outputs = model(clips)
        preds = torch.argmax(outputs, dim=1)

        for pred, start_idx in zip(preds.cpu(), starts):
            start = start_idx.item()
            end = start + dataset.chunk_frames
            frame_ranges[pred.item()].append((start, end))

# OpenCV video writer settings
cap = cv2.VideoCapture(match_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Save classified clips
for class_idx, ranges in frame_ranges.items():
    out_path = os.path.join(output_dir, f"{class_names[class_idx]}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for start, end in ranges:
        for f in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize if needed (safety)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)  # stays in BGR

    writer.release()
    print(f"Saved class {class_idx} video to: {out_path}")

cap.release()
