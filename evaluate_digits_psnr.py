import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------- Optical Utilities ----------
def random_phase_mask(shape):
    phase = 2 * np.pi * np.random.rand(*shape)
    return np.exp(1j * phase)

def angular_spectrum_propagation(u0, distance, wavelength, dx):
    N, M = u0.shape
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    k = 2 * np.pi / wavelength
    H = np.exp(1j * k * distance * np.sqrt(np.maximum(0, 1 - (wavelength * FX)**2 - (wavelength * FY)**2)))
    U0 = np.fft.fft2(u0)
    U1 = U0 * H
    u1 = np.fft.ifft2(U1)
    return u1

# ---------- UNet ----------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNet, self).__init__()
        self.enc0 = self.conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = self.conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(base_channels * 2, base_channels * 4)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec1 = self.conv_block(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = self.conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool1(x0))
        x2 = self.enc2(self.pool2(x1))
        x3 = self.up1(x2)
        x3 = torch.cat([x3, x1], dim=1)
        x3 = self.dec1(x3)
        x4 = self.up2(x3)
        x4 = torch.cat([x4, x0], dim=1)
        x4 = self.dec2(x4)
        return self.out_conv(x4)

# ---------- Evaluation ----------
def evaluate_digits(model, device, distances, args):
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    images = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy()

    psnr_results = {d: [] for d in range(10)}  # {digit: [ [psnr@d1,...], [psnr@d2,...] ]}
    dx = args.pixel_size

    for digit in range(10):
        digit_indices = np.where(labels == digit)[0][:args.samples_per_digit]
        digit_imgs = images[digit_indices]
        digit_scores = []

        for img in tqdm(digit_imgs, desc=f"Digit {digit}"):
            img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
            img_up = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
            u0 = img_up * random_phase_mask(img_up.shape)

            scores = []
            for d in distances:
                Uz = angular_spectrum_propagation(u0, d, args.wavelength, dx)
                blurred = np.abs(Uz) ** 2
                x = torch.tensor(blurred, dtype=torch.float32)[None, None, ...].to(device)

                with torch.no_grad():
                    for _ in range(args.steps - 1):
                        x = model(x)
                restored = x.cpu().numpy()[0, 0]
                scores.append(psnr_metric(img_up, restored, data_range=1.0))
            digit_scores.append(scores)

        psnr_results[digit] = np.mean(digit_scores, axis=0)

    return psnr_results

def plot_results(psnr_results, distances):
    plt.figure(figsize=(10, 6))
    for digit, scores in psnr_results.items():
        plt.plot(distances, scores, marker='o', label=f'Digit {digit}')
    plt.xlabel("Propagation Distance (m)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Distance for each Digit")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_distance.png", dpi=600)
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained .pth model")
    parser.add_argument('--steps', type=int, default=10, help="Number of restoration steps")
    parser.add_argument('--samples_per_digit', type=int, default=20, help="Samples per digit")
    parser.add_argument('--wavelength', type=float, default=5.32e-7)
    parser.add_argument('--pixel_size', type=float, default=1e-5)
    parser.add_argument('--min_distance', type=float, default=0.001)
    parser.add_argument('--max_distance', type=float, default=0.02)
    parser.add_argument('--num_distances', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    distances = np.linspace(args.min_distance, args.max_distance, args.num_distances)
    results = evaluate_digits(model, device, distances, args)
    plot_results(results, distances)
