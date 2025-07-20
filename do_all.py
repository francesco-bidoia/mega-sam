import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd):
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-vf",
        "fps=0.8",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    run_cmd(cmd, cwd=os.getcwd())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', '-s', required=True, type=str)
    parser.add_argument('--clean', '-c', action='store_true',
                        help='clean generated data and rerun all steps')
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    source_path = Path(args.source_path).resolve()
    mp4_files = glob.glob(str(source_path / '*.mp4'))
    if not mp4_files:
        raise FileNotFoundError(f'No mp4 file found in {source_path}')
    video_path = mp4_files[0]
    frames_dir = source_path / 'images'
    if args.clean and frames_dir.exists():
        shutil.rmtree(frames_dir)

    if not frames_dir.exists():
        extract_frames(video_path, frames_dir)

    depth_out = source_path / 'mono_depth'
    if args.clean and depth_out.exists():
        shutil.rmtree(depth_out)

    if not depth_out.exists():
        run_cmd([
            'python', 'Depth-Anything/run_videos.py',
            '--encoder', 'vitl',
            '--load-from', 'Depth-Anything/checkpoints/depth_anything_vitl14.pth',
            '--img-path', str(frames_dir),
            '--outdir', str(depth_out)
        ], cwd=root_dir)

    unidepth_out = source_path / 'unidepth'
    if args.clean and unidepth_out.exists():
        shutil.rmtree(unidepth_out)

    if not unidepth_out.exists():
        run_cmd([
            'python', 'UniDepth/scripts/demo_mega-sam.py',
            '--img-path', str(frames_dir),
            '--outdir', str(unidepth_out)
        ], cwd=root_dir)

    recon_dir = source_path / 'reconstructions'
    if args.clean and recon_dir.exists():
        shutil.rmtree(recon_dir)

    droid_out = source_path / 'outputs' / 'droid.npz'
    if args.clean and droid_out.exists():
        droid_out.unlink()

    if not recon_dir.exists() or not droid_out.exists():
        run_cmd([
            'python', 'camera_tracking_scripts/test_demo.py',
            '--datapath', str(frames_dir),
            '--weights', 'checkpoints/megasam_final.pth',
            '--outdir', str(source_path),
            '--mono_depth_path', str(source_path / 'mono_depth'),
            '--metric_depth_path', str(source_path / 'unidepth'),
        ], cwd=root_dir)

    cache_dir = source_path / 'cache_flow'
    if args.clean and cache_dir.exists():
        shutil.rmtree(cache_dir)

    if not cache_dir.exists():
        run_cmd([
            'python', 'cvd_opt/preprocess_flow.py',
            '--datapath', str(frames_dir),
            '--model', 'cvd_opt/raft-things.pth',
            '--output_dir', str(cache_dir),
            '--mixed_precision'
        ], cwd=root_dir)

    cvd_npz = source_path / 'outputs_cvd' / 'sgd_cvd_hr.npz'
    if args.clean and cvd_npz.exists():
        cvd_npz.unlink()

    if not cvd_npz.exists():
        run_cmd([
            'python', 'cvd_opt/cvd_opt.py',
            '--w_grad', '2.0',
            '--w_normal', '5.0',
            '--output_dir', str(source_path / 'cvd_output'),
            '--recon_dir', str(source_path / 'reconstructions'),
            '--cache_dir', str(source_path / 'cache_flow')
        ], cwd=root_dir)

    if cvd_npz.exists():
        npz_for_colmap = cvd_npz
    else:
        npz_for_colmap = droid_out

    # Export final results to COLMAP format
    colmap_out = root_dir / 'colmap_temp'
    run_cmd([
        'python', 'export_to_colmap.py',
        '--npz', str(npz_for_colmap),
        '--frames', str(frames_dir),
        '--outdir', str(colmap_out)
    ], cwd=root_dir)

    sparse_src = colmap_out / 'sparse'
    if sparse_src.exists():
        dst = source_path / 'sparse'
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(sparse_src), dst)
    shutil.rmtree(colmap_out, ignore_errors=True)


if __name__ == '__main__':
    main()
