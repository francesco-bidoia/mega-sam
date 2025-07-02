import argparse
import glob
import os
import shutil
import subprocess
import cv2
from pathlib import Path


def run_cmd(cmd, cwd):
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', '-s', required=True, type=str)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    source_path = Path(args.source_path).resolve()
    mp4_files = glob.glob(str(source_path / '*.mp4'))
    if not mp4_files:
        raise FileNotFoundError(f'No mp4 file found in {source_path}')
    video_path = mp4_files[0]
    scene_name = Path(video_path).stem

    frames_dir = source_path / 'frames' / scene_name
    extract_frames(video_path, frames_dir)

    depth_out = source_path / 'mono_depth' / scene_name
    run_cmd([
        'python', 'Depth-Anything/run_videos.py',
        '--encoder', 'vitl',
        '--load-from', 'Depth-Anything/checkpoints/depth_anything_vitl14.pth',
        '--img-path', str(frames_dir),
        '--outdir', str(depth_out)
    ], cwd=root_dir)

    run_cmd([
        'python', 'UniDepth/scripts/demo_mega-sam.py',
        '--scene-name', scene_name,
        '--img-path', str(frames_dir),
        '--outdir', str(source_path / 'unidepth')
    ], cwd=root_dir)

    run_cmd([
        'python', 'camera_tracking_scripts/test_demo.py',
        '--datapath', str(frames_dir),
        '--weights', 'checkpoints/megasam_final.pth',
        '--scene_name', scene_name,
        '--mono_depth_path', str(source_path / 'mono_depth'),
        '--metric_depth_path', str(source_path / 'unidepth'),
        '--disable_vis'
    ], cwd=root_dir)

    run_cmd([
        'python', 'cvd_opt/preprocess_flow.py',
        '--datapath', str(frames_dir),
        '--model', 'cvd_opt/raft-things.pth',
        '--scene_name', scene_name,
        '--mixed_precision'
    ], cwd=root_dir)

    run_cmd([
        'python', 'cvd_opt/cvd_opt.py',
        '--scene_name', scene_name,
        '--w_grad', '2.0',
        '--w_normal', '5.0',
        '--output_dir', str(source_path / 'cvd_output')
    ], cwd=root_dir)

    # Move generated folders to source_path
    recon_src = root_dir / 'reconstructions' / scene_name
    if recon_src.exists():
        dst = source_path / 'reconstructions'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(recon_src), dst / scene_name)

    cache_src = root_dir / 'cache_flow' / scene_name
    if cache_src.exists():
        dst = source_path / 'cache_flow'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cache_src), dst / scene_name)

    out_file = root_dir / 'outputs' / f'{scene_name}_droid.npz'
    if out_file.exists():
        dst = source_path / 'outputs'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_file), dst / out_file.name)

    cvd_file = root_dir / 'outputs_cvd' / f'{scene_name}_sgd_cvd_hr.npz'
    if cvd_file.exists():
        dst = source_path / 'outputs_cvd'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cvd_file), dst / cvd_file.name)


if __name__ == '__main__':
    main()
