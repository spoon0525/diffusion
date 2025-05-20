import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_imgs = dataset.data.numpy().astype(np.float32) / 255.0

    indices = np.where(dataset.targets.numpy() == args.digit)[0]
    if len(indices) == 0:
        print(f"No image found for digit {args.digit}")
        return

    test_img = test_imgs[indices[0]]
    img = torch.tensor(test_img).unsqueeze(0).unsqueeze(0)
    test_img_up = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()

    u0 = test_img_up * random_phase_mask(test_img_up.shape)
    Uz = angular_spectrum_propagation(u0, args.distance, args.wavelength, args.pixel_size)
    blurred = np.abs(Uz) ** 2

    x = torch.tensor(blurred, dtype=torch.float32)[None, None, ...].to(device)
    restored_steps = [blurred]

    with torch.no_grad():
        for _ in range(args.steps - 1):
            x = model(x)
            restored_steps.append(x.cpu().numpy()[0, 0])

    restored_final = restored_steps[-1]
    psnr_val = psnr_metric(test_img_up, restored_final, data_range=1.0)
    ssim_val = ssim_metric(test_img_up, restored_final, data_range=1.0)

    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    fig, axs = plt.subplots(1, 6, figsize=(20, 4), constrained_layout=True)
    axs[0].imshow(test_img_up, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    step_indices = np.linspace(0, len(restored_steps)-1, 5, dtype=int)
    for i, idx in enumerate(step_indices):
        axs[i+1].imshow(restored_steps[idx], cmap='gray')
        if idx == 0:
            axs[i+1].set_title("Blurred (Input)", fontsize=14)
        elif idx == len(restored_steps) - 1:
            axs[i+1].set_title("Final Restored", fontsize=14)
        else:
            axs[i+1].set_title(f"Step {idx}", fontsize=14)
        axs[i+1].axis('off')
    plt.savefig(f"result\\{args.model_path.split('\\')[1]}\\{round(args.distance*100, 1) if args.distance*100%1 != 0 else int(args.distance*100)}cm\\{args.digit}_{round(args.distance*100, 1) if args.distance*100%1 != 0 else int(args.distance*100)}cm_psnr{psnr_val:.2f}.png", dpi=600)
    #plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model .pth file")
    parser.add_argument('--digit', type=int, default=0, help="Digit to test (0-9)")
    parser.add_argument('--distance', type=float, default=0.01, help="Propagation distance in meters")
    parser.add_argument('--steps', type=int, default=10, help="Number of restoration steps")
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help="Wavelength in meters")
    parser.add_argument('--pixel_size', type=float, default=1e-5, help="Pixel size in meters")
    args = parser.parse_args()

    test(args)
