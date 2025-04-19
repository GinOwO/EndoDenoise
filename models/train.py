import argparse
import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from typing import List


class DnCNN(nn.Module):
    def __init__(self, channels: int = 1, num_of_layers: int = 17) -> None:
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, 3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.dncnn(x)


class VideoDenoiseDataset(Dataset):
    def __init__(
        self, video_paths: List[str], transform: transforms.Compose | None = None
    ) -> None:
        self.video_paths = video_paths
        self.transform = transform

    def _extract_frames(self, path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        return frames

    def __len__(self) -> int:
        return sum(len(self._extract_frames(path)) for path in self.video_paths)

    def __getitem__(self, idx: int):
        video_idx = 0
        while idx >= len(self._extract_frames(self.video_paths[video_idx])):
            idx -= len(self._extract_frames(self.video_paths[video_idx]))
            video_idx += 1
        clean = (
            self._extract_frames(self.video_paths[video_idx])[idx].astype(np.float32)
            / 255.0
        )
        noise = np.random.normal(0, 0.1, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0.0, 1.0)
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
        return noisy, clean


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    video_files = [
        os.path.join(args.dataset_path, f)
        for f in os.listdir(args.dataset_path)
        if f.endswith(".mp4")
    ]
    random.seed(args.seed)
    selected_videos = random.sample(video_files, args.n_samples)

    dataset = VideoDenoiseDataset(selected_videos, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        ):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}] Loss: {epoch_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory containing .mp4 videos",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to save the trained model (.pth)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of video samples to randomly select for training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    train(args)
