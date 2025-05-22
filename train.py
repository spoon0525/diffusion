import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_metric, peak_signal_noise_ratio as psnr_metric
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def random_phase(shape):
    phase = 2 * np.pi * np.random.rand(*shape)
    return np.exp(1j * phase)

def angular_spectrum_propagation(u0, distance, wavelength, dx):
    N, M = u0.shape
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    k = 2 * np.pi / wavelength
    H = np.exp(1j * k * distance * np.sqrt(np.maximum(0, 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2)))
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
def combined_loss(pred, target):
    mse = F.mse_loss(pred, target)
    ssim_val = 0
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for i in range(pred_np.shape[0]):
        ssim_val += 1 - ssim_metric(target_np[i, 0], pred_np[i, 0], data_range=1.0)
    ssim_val /= pred_np.shape[0]
    return mse + ssim_val

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    images = dataset.data[:args.samples].numpy().astype(np.float32) / 255.0

    upsampled_images = []
    for img in images:
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
        upsampled = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        upsampled_images.append(upsampled.squeeze().numpy())
    images = np.array(upsampled_images)

    distances = np.linspace(0, args.max_distance, num=args.blur_levels)
    wavelength = args.wavelength
    dx = args.pixel_size

    train_inputs, train_targets = [], []
    for img in tqdm(images, desc="Simulating Optical Propagation"):
        u0 = img * random_phase(img.shape)
        blurred_seq = [np.abs(angular_spectrum_propagation(u0, z, wavelength, dx))**2 for z in distances]
        for t in range(1, args.blur_levels):
            train_inputs.append(blurred_seq[t].astype(np.float32))
            train_targets.append(blurred_seq[t - 1].astype(np.float32))

    train_inputs = torch.tensor(np.array(train_inputs))[:, None, ...].to(device)
    train_targets = torch.tensor(np.array(train_targets))[:, None, ...].to(device)

    dataset = TensorDataset(train_inputs, train_targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr = 0
    psnr_history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_x, batch_y in pbar:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = combined_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(dataloader)
        psnr_val = evaluate_psnr(model, device, train_inputs, train_targets, args.batch_size)
        psnr_history.append(psnr_val)
        model_path = f"result\\{round(args.max_distance*100, 1) if args.max_distance*100%1 != 0 else int(args.max_distance*100)}cm\\model\\best.pth"
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, PSNR: {psnr_val:.2f} dB")

    plt.figure(figsize=(8, 6))
    plt.plot(psnr_history, marker='o')
    plt.title('PSNR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid()
    plt.savefig(f"result\\{round(args.max_distance*100, 1) if args.max_distance*100%1 != 0 else int(args.max_distance*100)}cm\\psnr_curve.png", dpi=600)
    plt.close()

    visualize_restoration(model, device, args, distances, wavelength, dx)
def evaluate_psnr(model, device, inputs, targets, batch_size=32):
    model.eval()
    psnr_vals = []
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            batch_x = inputs[i:i+batch_size]
            batch_y = targets[i:i+batch_size]
            outputs = model(batch_x)
            preds = outputs.cpu().numpy()
            targets_np = batch_y.cpu().numpy()
            for j in range(preds.shape[0]):
                psnr_vals.append(psnr_metric(targets_np[j, 0], preds[j, 0], data_range=1.0))
    return np.mean(psnr_vals)

def visualize_restoration(model, device, args, distances, wavelength, dx):
    import matplotlib.gridspec as gridspec
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_img = test_dataset.data[0].numpy().astype(np.float32) / 255.0

    img = torch.tensor(test_img).unsqueeze(0).unsqueeze(0)
    test_img_up = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()

    u0 = test_img_up * random_phase(test_img_up.shape)
    Uz = angular_spectrum_propagation(u0, args.max_distance, wavelength, dx)
    blurred = np.abs(Uz) ** 2

    x = torch.tensor(blurred, dtype=torch.float32)[None, None, ...].to(device)
    model.eval()
    restored_steps = [blurred]

    with torch.no_grad():
        for _ in range(args.blur_levels - 1):
            x = model(x)
            restored_steps.append(x.cpu().numpy()[0, 0])

    total_steps = len(restored_steps)
    indices = np.linspace(0, total_steps - 1, 5, dtype=int)

    fig = plt.figure(figsize=(20, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 6, figure=fig)

    ax = plt.subplot(gs[0, 0])
    ax.imshow(test_img_up, cmap='gray')
    ax.set_title("Original", fontsize=14)
    ax.axis('off')

    for i, idx in enumerate(indices):
        ax = plt.subplot(gs[0, i + 1])
        ax.imshow(restored_steps[idx], cmap='gray')
        if idx == 0:
            ax.set_title("Blurred (Input)", fontsize=14)
        elif idx == total_steps - 1:
            ax.set_title("Final Restored", fontsize=14)
        else:
            ax.set_title(f"Step {idx}", fontsize=14)
        ax.axis('off')

    plt.savefig(f"result\\{round(args.max_distance*100, 1) if args.max_distance*100%1 != 0 else int(args.max_distance*100)}cm\\restoration_steps.png", dpi=600)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000, help="Number of MNIST images to use")
    parser.add_argument('--blur_levels', type=int, default=20, help="Number of blur levels (diffusion steps)")
    parser.add_argument('--max_distance', type=float, default=0.01, help="Max propagation distance in meters")
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help="Wavelength in meters")
    parser.add_argument('--pixel_size', type=float, default=1e-5, help="Pixel size in meters")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train(args)