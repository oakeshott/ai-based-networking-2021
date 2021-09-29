from skimage.metrics import structural_similarity
import cv2
import os
import glob
import json
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description="Preprocessing")
parser.add_argument('--original', default='.',
        help='Input data directory')
parser.add_argument('--received', default='.',
        help='Input data directory')
parser.add_argument('-o', '--output', default='.',
        help='Output data directory')
parser.add_argument('--basename', default='frame',
        help='Basename')
parser.add_argument('-j', '--n-jobs', default=1, type=int,
        help='Number of jobs')
parser.add_argument('--test-data', action="store_true",
        help='active flag if preprocessing test data')
parser.add_argument('--grayscale', action="store_true",
        help='Grayscale')

args = parser.parse_args()

def calc_similarity_measures(src, dst, grayscale=False):
    if grayscale:
        src_img = cv2.imread(src, cv2.COLOR_BGR2GRAY)
        dst_img = cv2.imread(dst, cv2.COLOR_BGR2GRAY)
        ssim = structural_similarity(src_img, dst_img)
    else:
        src_img = cv2.imread(src)
        dst_img = cv2.imread(dst)
        ssim = structural_similarity(src_img, dst_img, multichannel=True)
    psnr = cv2.PSNR(src_img, dst_img)
    return ssim, psnr

def calc_all_frame_similarity(org_path, rev_path, rev_dir, dst_dir, grayscale=False):
    split_list = rev_dir.split("_")
    throughput = split_list[-2]
    loss_rate  = split_list[-1]
    throughput = int(throughput[0:4])
    loss_rate  = float(loss_rate[0] + '.' + loss_rate[1:])
    if len(split_list) == 3:
        org_dir = split_list[0]
    else:
        org_dir = split_list[0] + "_" + split_list[1]

    org_frame_files = glob.glob(os.path.join(os.path.join(org_path, org_dir), '*'))
    rev_frame_files = glob.glob(os.path.join(os.path.join(rev_path, rev_dir), '*'))

    org_frame_files.sort()
    rev_frame_files.sort()

    org_frame_size = len(org_frame_files)
    rev_frame_size = len(rev_frame_files)
    diff = org_frame_size - rev_frame_size

    org_frame_files = org_frame_files[:rev_frame_size]
    filename = os.path.join(dst_dir, f"similarity_{rev_dir}.json")
    rets = []
    for frame_idx, (org_frame_file, rev_frame_file) in enumerate(zip(org_frame_files, rev_frame_files)):
        ssim, psnr = calc_similarity_measures(org_frame_file, rev_frame_file, grayscale)
        ret = {
                "video_type": org_dir,
                "loss_rate": loss_rate,
                "throughput": throughput,
                "frame_index": frame_idx,
                "ssim": ssim,
                "psnr": psnr
                }
        rets.append(ret)
    with open(filename, "w") as f:
        json.dump(rets, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def call_calc_all_frame_similarity_func(org_path, rev_path, rev_dirs, dst_dir, grayscale=False):
    for rev_dir in tqdm(rev_dirs):
        calc_all_frame_similarity(org_path, rev_path, rev_dir, dst_dir, grayscale)

def split_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def preprocessing_train_data():
    org_path = args.original
    rev_path = args.received
    dst_dir  = args.output
    n_jobs   = args.n_jobs
    org_dirs = os.listdir(org_path)
    rev_dirs = os.listdir(rev_path)
    org_dirs.sort()
    rev_dirs.sort()

    li_rev_dirs = []

    for org_dir in org_dirs:
        tmp = []
        for rev_dir in rev_dirs:
            if org_dir in rev_dir:
                tmp.append(rev_dir)
        li_rev_dirs.append(tmp)

    for _rev_dirs in li_rev_dirs:
        os.makedirs(dst_dir, exist_ok=True)

        jobs = list(split_list(_rev_dirs, len(_rev_dirs) // n_jobs))
        Parallel(n_jobs=n_jobs)(delayed(call_calc_all_frame_similarity_func)(org_path, rev_path, rev_dirs, dst_dir, args.grayscale) for rev_dirs in jobs)

def preprocessing_test_data():
    org_path = args.original
    rev_path = args.received
    dst_dir  = args.output

    org_dir = "."
    rev_dir = "."
    throughput = 0
    loss_rate  = 0

    os.makedirs(dst_dir, exist_ok=True)

    org_frame_files = glob.glob(os.path.join(os.path.join(org_path, org_dir), '*'))
    rev_frame_files = glob.glob(os.path.join(os.path.join(rev_path, rev_dir), '*'))

    org_frame_files.sort()
    rev_frame_files.sort()

    org_frame_size = len(org_frame_files)
    rev_frame_size = len(rev_frame_files)
    diff = org_frame_size - rev_frame_size
    idx = rev_path.split("/")[-1]
    org_frame_files = org_frame_files[:rev_frame_size]
    filename = os.path.join(dst_dir, f"similarity_{idx}.json")
    rets = []
    for frame_idx, (org_frame_file, rev_frame_file) in tqdm(enumerate(zip(org_frame_files, rev_frame_files))):
        ssim, psnr = calc_similarity_measures(org_frame_file, rev_frame_file, args.grayscale)
        ret = {
                "video_type": f"test_{idx}",
                "loss_rate": loss_rate,
                "throughput": throughput,
                "frame_index": frame_idx,
                "ssim": ssim,
                "psnr": psnr
                }
        rets.append(ret)
    with open(filename, "w") as f:
        json.dump(rets, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def main():
    if args.test_data:
        preprocessing_test_data()
    else:
        preprocessing_train_data()

if __name__ == '__main__':
    main()
