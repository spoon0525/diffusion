import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--start_distance', type=float, default=0.01, help='Begin of distance range for test image in meter')
parser.add_argument('--end_distance', type=float, default=0.02, help='End of distance range for test image in meter')
args=parser.parse_args()

for d in np.arange(args.start_distance, args.end_distance+0.001, 0.001):
    for num in range(10):
        os.system(f'python test.py --model_path=result\\1.5cm\\model\\model_epoch10_psnr17.40.pth --digit={num} --distance={round(d, 3)}')