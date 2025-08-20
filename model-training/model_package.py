import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision.transforms as T
import torch.nn.functional as F
from gluoncv.torch.model_zoo import get_model
from gluoncv.torch.engine.config import get_cfg_defaults
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video, _, _ = read_video(self.video_paths[idx], pts_unit='sec')  # (T, H, W, C) output shape from torchvision
        video = video.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W) to match PyTorch format

        video = uniform_sampling(video, 32) # Uniformly sample frames

        if self.transform:
            video = self.transform(video)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return video, label


class VideoDataset2(Dataset):
    def __init__(self, video_paths, labels, chunk_size=16, transform=None):
        self.samples = []  # Will store tuples: (video_path, start_frame, label)
        self.chunk_size = chunk_size
        self.transform = transform

        for path, label in zip(video_paths, labels):
            video, _, _ = read_video(path, pts_unit='sec')  # (T, H, W, C)
            total_frames = video.shape[0]

            # Precompute start indices for full chunks only
            for start in range(0, total_frames - chunk_size + 1, chunk_size):
                self.samples.append((path, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start_frame, label = self.samples[idx]
        
        video, _, _ = read_video(path, pts_unit='sec')  # Load full video
        video = video[start_frame:start_frame + self.chunk_size]  # (chunk_size, H, W, C)
        video = video.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)

        if self.transform:
            video = self.transform(video)

        label = torch.tensor(label, dtype=torch.long)
        return video, label


class VideoChunkDataset(Dataset):
    def __init__(self, video_paths, labels, chunk_size=64, sample_frames=32, stride=16, transform=None):
        self.samples = []
        self.chunk_size = chunk_size
        self.sample_frames = sample_frames
        self.stride = stride
        self.transform = transform

        for path, label in zip(video_paths, labels):
            video, _, _ = read_video(path, pts_unit='sec')
            total_frames = video.shape[0]

            # Sliding-window: move forward by 'stride' each time
            for start in range(0, total_frames - chunk_size + 1, stride):
                self.samples.append((path, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start_frame, label = self.samples[idx]
        
        video, _, _ = read_video(path, pts_unit='sec')
        video = video[start_frame:start_frame + self.chunk_size]  # (T, H, W, C)
        video = video.permute(3, 0, 1, 2).float() / 255.0          # (C, T, H, W)

        video = uniform_sampling(video, self.sample_frames) # Uniformly sample frames

        if self.transform:
            video = self.transform(video)

        return video, torch.tensor(label, dtype=torch.long)


class MatchChunkDataset(Dataset):
    def __init__(self, video_path, chunk_frames=64, sample_frames=32, resize=(224, 224)):
        self.video_path = video_path
        self.chunk_frames = chunk_frames
        self.sample_frames = sample_frames
        self.resize = resize

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.clip_start = list(range(0, self.total_frames - chunk_frames + 1, chunk_frames))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(resize)
        ])

    def __len__(self):
        return len(self.clip_start)

    def __getitem__(self, idx):
        start_idx = self.clip_start[idx]
        frames = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(self.chunk_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

        # Uniformly sample frames
        step = self.chunk_frames // self.sample_frames
        sampled = [frames[i] for i in range(0, self.chunk_frames, step)]
        clip_tensor = torch.stack(sampled).permute(1, 0, 2, 3)  # (3, 16, H, W)

        return clip_tensor, start_idx
    

def uniform_sampling(video, num_samples):
    T = video.shape[1]
    if T == num_samples:
        return video
    elif T > num_samples:
        indices = torch.linspace(0, T - 1, steps=num_samples).long() # Sample indices evenly across the video
        return video[:, indices, :, :]
    else:
        last = video[:, -1:, :, :].repeat(1, num_samples - T, 1, 1) # Pad by repeating last frame
        return torch.cat([video, last], dim=1)
    

def create_paths_labels(file_paths):
    video_paths = []
    labels = []
    
    if isinstance(file_paths, str):
        file_paths = [file_paths] # Ensure file_paths is a list

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    path, label = line.rsplit(' ', 1)
                    video_paths.append(path)
                    labels.append(int(label))
                    
    return video_paths, labels


def resize_tensor(video, size=(224, 224)):
    video = video.permute(1, 0, 2, 3) # [C, T, H, W] → [T, C, H, W] for interpolation
    video = F.interpolate(video, size=size, mode='bilinear', align_corners=False) # Resize each frame using interpolate
    
    return video.permute(1, 0, 2, 3) # Back to [C, T, H, W]


def normalize_clip(x):
    # x is (C,T,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)

    return (x - mean) / std


def load_model(model_name='badminton-rally-classification/model-training/models/i3d_resnet50_v1_kinetics400.yaml', num_classes=5):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(model_name)
    model = get_model(cfg)
    
    # Replace classification head (usually 'head' is used in I3D)
    if hasattr(model, 'head'):
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    # If for some reason 'fc' is used (less likely), fallback
    elif hasattr(model, 'fc'):
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    return model.to(device)


def update_plot(epoch, train_losses, val_accuracies, num_epochs):
    clear_output(wait=True)
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(len(train_losses)))
    plt.legend()

    plt.subplot(2, 1, 2) 
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(len(val_accuracies)))
    plt.legend()

    plt.suptitle(f"Epoch {epoch}/{num_epochs}")
    plt.tight_layout()
    plt.show()