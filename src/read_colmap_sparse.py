#!/usr/bin/env python3
import os
import sys
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---- Basic type definitions -------------------------------------------------

@dataclass
class CameraModel:
    model_id: int
    model_name: str
    num_params: int


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class Image:
    id: int
    qvec: Tuple[float, float, float, float]   # (qw, qx, qy, qz)
    tvec: Tuple[float, float, float]         # (tx, ty, tz)
    camera_id: int
    name: str
    xys: List[Tuple[float, float]]           # 2D keypoints
    point3D_ids: List[int]                   # corresponding POINT3D_IDs (or -1)


# Camera models used by COLMAP (id -> name, #params) :contentReference[oaicite:1]{index=1}
CAMERA_MODELS = [
    CameraModel(0, "SIMPLE_PINHOLE",        3),
    CameraModel(1, "PINHOLE",              4),
    CameraModel(2, "SIMPLE_RADIAL",        4),
    CameraModel(3, "RADIAL",               5),
    CameraModel(4, "OPENCV",               8),
    CameraModel(5, "OPENCV_FISHEYE",       8),
    CameraModel(6, "FULL_OPENCV",          12),
    CameraModel(7, "FOV",                  5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE",       5),
    CameraModel(10, "THIN_PRISM_FISHEYE",  12),
]

CAMERA_MODEL_IDS: Dict[int, CameraModel] = {m.model_id: m for m in CAMERA_MODELS}


# ---- Helpers ----------------------------------------------------------------

def read_next_bytes(fid, num_bytes: int, fmt: str):
    """
    Read and unpack the next bytes from a binary file using little endian.
    fmt is a struct format string WITHOUT the leading '<'.
    """
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of file.")
    return struct.unpack("<" + fmt, data)


# ---- Read cameras.bin -------------------------------------------------------

def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    Read COLMAP cameras.bin.
    Format (little endian):
      uint64 num_cameras
      for each camera:
        int32   camera_id
        int32   model_id
        uint64  width
        uint64  height
        double[num_params] params
    :contentReference[oaicite:2]{index=2}
    """
    cameras: Dict[int, Camera] = {}

    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            cam_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")

            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"Unknown camera model_id {model_id}")

            model = CAMERA_MODEL_IDS[model_id]
            num_params = model.num_params

            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))

            cameras[cam_id] = Camera(
                id=cam_id,
                model=model.model_name,
                width=width,
                height=height,
                params=params,
            )

    return cameras


# ---- Read images.bin --------------------------------------------------------

def read_images_binary(path: str) -> List[Image]:
    """
    Read COLMAP images.bin.
    Format (little endian):
      uint64 num_reg_images
      for each image:
        int32   image_id
        double  qw, qx, qy, qz
        double  tx, ty, tz
        int32   camera_id
        char[]  image_name (null-terminated)
        uint64  num_points2D
        repeated num_points2D times:
          double x, double y, int64 point3D_id
    :contentReference[oaicite:3]{index=3}
    """
    images: List[Image] = []

    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_reg_images):
            # 1 * int32 + 7 * double + 1 * int32 = 4 + 56 + 4 = 64 bytes
            vals = read_next_bytes(fid, 64, "idddddddi")

            image_id = vals[0]
            qvec = tuple(vals[1:5])          # qw, qx, qy, qz
            tvec = tuple(vals[5:8])          # tx, ty, tz
            camera_id = vals[8]

            # Read null-terminated image name
            name_bytes = []
            while True:
                c = read_next_bytes(fid, 1, "c")[0]
                if c == b"\x00":
                    break
                name_bytes.append(c)

            image_name = b"".join(name_bytes).decode("utf-8")

            # 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            if num_points2D > 0:
                # each point: double x, double y, int64 point3D_id -> 24 bytes
                raw = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = []
                point3D_ids = []
                for i in range(num_points2D):
                    x = raw[3 * i + 0]
                    y = raw[3 * i + 1]
                    pid = raw[3 * i + 2]
                    xys.append((x, y))
                    point3D_ids.append(int(pid))
            else:
                xys = []
                point3D_ids = []

            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )

    return images


# ---- Convenience: read both from a COLMAP sparse dir -----------------------

def read_colmap_sparse_model(model_dir: str):
    """
    model_dir should be something like:
      gs_data/Liza_gemeni/sparse/0
    containing cameras.bin and images.bin.
    """
    cameras_path = os.path.join(model_dir, "cameras.bin")
    images_path = os.path.join(model_dir, "images.bin")

    if not os.path.isfile(cameras_path):
        raise FileNotFoundError(f"Missing {cameras_path}")
    if not os.path.isfile(images_path):
        raise FileNotFoundError(f"Missing {images_path}")

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    return cameras, images


# ---- Simple CLI -------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_colmap_sparse.py <sparse_model_dir>")
        print("Example: python read_colmap_sparse.py gs_data/Liza_gemeni/sparse/0")
        sys.exit(1)

    model_dir = sys.argv[1]
    cameras, images = read_colmap_sparse_model(model_dir)

    print(f"Loaded {len(cameras)} cameras")
    for cid, cam in cameras.items():
        print(f"\nCamera {cid}:")
        print(f"  model  = {cam.model}")
        print(f"  size   = {cam.width} x {cam.height}")
        print(f"  params = {cam.params}")

    print(f"\nLoaded {len(images)} registered images")
    for img in images:
        print(f"\nImage {img.id}: {img.name}")
        print(f"  CAMERA_ID = {img.camera_id}")
        print(f"  qvec      = {img.qvec}")
        print(f"  tvec      = {img.tvec}")
        print(f"  #points2D = {len(img.xys)}")
        if len(img.xys) > 0:
            print(f"  First 3 points2D (x, y, POINT3D_ID):")
            for (x, y), pid in list(zip(img.xys, img.point3D_ids))[:3]:
                print(f"    ({x:.2f}, {y:.2f}) -> {pid}")

if __name__ == "__main__":
    main()
