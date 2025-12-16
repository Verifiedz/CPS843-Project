import argparse
from pathlib import Path

import cv2
import numpy as np

from src.models.dcp import dehaze_dcp_clahe


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def save_output(result_bgr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), result_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write output: {out_path}")


def process_one(input_path: Path, output_path: Path, args) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    result = dehaze_dcp_clahe(
        img_bgr_uint8=img,
        patch_size=args.patch_size,
        omega=args.omega,
        t0=args.t0,
        refine=not args.no_refine,
        use_clahe=not args.no_clahe,
        white_balance=not args.no_wb,
    )

    save_output(result, output_path)


def self_test() -> None:
    # Synthetic quick sanity check (no dataset needed)
    h, w = 240, 320
    clear = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(clear, (20, 20), (140, 200), (40, 200, 40), -1)
    cv2.circle(clear, (240, 120), 60, (200, 40, 40), -1)
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    clear[:, :, 2] = grad

    # Add synthetic haze: I = J*t + A*(1-t)
    A = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    t = 0.55
    clear_f = clear.astype(np.float32) / 255.0
    hazy_f = clear_f * t + A.reshape(1, 1, 3) * (1 - t)
    hazy = (np.clip(hazy_f, 0, 1) * 255).astype(np.uint8)

    out = dehaze_dcp_clahe(hazy, refine=True, use_clahe=True, white_balance=True)

    if out.dtype != np.uint8 or out.shape != hazy.shape:
        raise AssertionError("Self-test failed: bad output shape/dtype")

    print("[SELF-TEST] PASS (pipeline runs, output shape/dtype OK)")


def main():
    parser = argparse.ArgumentParser(description="Dev1: DCP + Reconstruction + CLAHE baseline")
    parser.add_argument("--input", "-i", required=True, help="Input image path OR folder of images")
    parser.add_argument("--output", "-o", required=True, help="Output image path OR output folder")

    # Dev1 knobs
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, default=0.1)

    parser.add_argument("--no-refine", action="store_true", help="Disable transmission refinement")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--no-wb", action="store_true", help="Disable white balance")

    parser.add_argument("--self-test", action="store_true", help="Run a quick synthetic sanity check and exit")

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Folder mode
    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in in_path.iterdir() if p.is_file() and is_image(p)])
        if not images:
            raise ValueError(f"No images found in: {in_path}")

        for p in images:
            out_file = out_path / f"DCP_result_{p.name}"
            print(f"[DCP] {p.name} -> {out_file.name}")
            process_one(p, out_file, args)

        print("[DCP] Done (folder).")
        return

    # Single image mode
    if out_path.is_dir() or str(args.output).endswith(("/", "\\")):
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"DCP_result_{in_path.name}"
    else:
        out_file = out_path

    print(f"[DCP] {in_path.name} -> {out_file}")
    process_one(in_path, out_file, args)
    print("[DCP] Done (single image).")


if __name__ == "__main__":
    main()
