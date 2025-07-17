import argparse
import numpy as np
import shutil
from pathlib import Path

from colmap_read_model import rotmat2qvec


def backproject(depth, K_inv):
    """Backproject depth map to camera space points."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    homog = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    rays = (K_inv @ homog.T).T
    return rays.reshape(h, w, 3) * depth[..., None]


def export(npz_path: Path, frames_dir: Path, out_dir: Path) -> None:
    data = np.load(npz_path)
    images = data['images']
    depths = data.get('depths')
    K = data['intrinsic']
    cam_c2w = data['cam_c2w']

    h, w = images.shape[1:3]
    K_inv = np.linalg.inv(K)
    out_dir = Path(out_dir)
    sparse_dir = out_dir / 'sparse'
    img_dir = out_dir / 'images'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for ext in ('*.jpg', '*.png', '*.jpeg'):
        frame_paths.extend(sorted(frames_dir.glob(ext)))
    if len(frame_paths) < cam_c2w.shape[0]:
        raise ValueError('Not enough frames in frames_dir')

    for i in range(cam_c2w.shape[0]):
        dst = img_dir / frame_paths[i].name
        shutil.copy(frame_paths[i], dst)

    with open(sparse_dir / 'cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')

    with open(sparse_dir / 'images.txt', 'w') as f_img, \
         open(sparse_dir / 'points3D.txt', 'w') as f_pts:
        f_img.write('# Image list with two lines of data per image:\n')
        f_img.write('#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, IMAGE_NAME\n')
        f_img.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')

        f_pts.write('# 3D point list with one line of data per point:\n')
        f_pts.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, ' \
                   'TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')

        point_id = 1
        stride = max(1, min(h, w) // 64)
        for i in range(cam_c2w.shape[0]):
            w2c = np.linalg.inv(cam_c2w[i])
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            fname = frame_paths[i].name
            f_img.write(
                f'{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} '
                f'{tvec[0]} {tvec[1]} {tvec[2]} 1 {fname}\n'
            )

            pts2d = []
            if depths is not None:
                cam_points = backproject(depths[i], K_inv)
                world_points = (
                    cam_c2w[i][:3, :3] @ cam_points.reshape(-1, 3).T
                ).T + cam_c2w[i][:3, 3]
                world_points = world_points.reshape(h, w, 3)
                idx = 0
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        z = depths[i, y, x]
                        if z <= 0:
                            continue
                        xyz = world_points[y, x]
                        r, g, b = images[i, y, x]
                        f_pts.write(
                            f'{point_id} {xyz[0]} {xyz[1]} {xyz[2]} '
                            f'{int(r)} {int(g)} {int(b)} 0 {i+1} {idx}\n'
                        )
                        pts2d.append(f'{x} {y} {point_id}')
                        point_id += 1
                        idx += 1
            f_img.write(' '.join(pts2d) + '\n\n')


def main():
    parser = argparse.ArgumentParser(description='Convert MegaSaM output to COLMAP format')
    parser.add_argument('--npz', required=True, type=Path, help='Path to .npz result file')
    parser.add_argument('--frames', required=True, type=Path, help='Directory with extracted frames')
    parser.add_argument('--outdir', required=True, type=Path, help='Destination directory for COLMAP files')
    args = parser.parse_args()
    export(args.npz, args.frames, args.outdir)


if __name__ == '__main__':
    main()


