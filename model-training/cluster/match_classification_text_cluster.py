import os
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_package import load_model, MatchChunkDataset
from tqdm import tqdm

# Paths
output_dir = 'badminton-rally-classification/model-training/output_pth4169_text'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "match_annotated.mp4")

model_path = 'badminton-rally-classification/model-training/trained_models/trained_model_4.pth'
match_path = 'badminton-rally-classification/dataset-creation/match-videos/match_9-sb.mp4'

class_names = ['interval', 'rally', 'shuttle change', 'floor mopping', 'set break']

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(model_path, map_location=device)
model = load_model(num_classes=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Dataset
dataset = MatchChunkDataset(match_path, chunk_frames=16, sample_frames=16)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Collect predictions for all frames
frame_predictions = {}  # frame_idx -> predicted class

with torch.no_grad():
    for clips, starts in tqdm(loader, desc="Classifying..."):
        clips = clips.to(device)
        outputs = model(clips)
        preds = torch.argmax(outputs, dim=1).cpu()

        for pred, start_idx in zip(preds, starts):
            start = start_idx.item()
            end = start + dataset.chunk_frames
            for f in range(start, end):
                frame_predictions[f] = pred.item()

# Read original video and write annotated video
cap = cv2.VideoCapture(match_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get predicted class for this frame
    pred_class = class_names[frame_predictions.get(frame_idx, 0)]  # default 'interval'
    
    # Put label text on top-right corner
    cv2.putText(frame, pred_class, (width-250, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(frame)
    frame_idx += 1

writer.release()
cap.release()
print(f"✅ Saved annotated video to: {output_path}")
