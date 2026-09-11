"""Convert an extracted Sentinel-2 SAFE product into HR training chips."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.windows import Window, transform as window_transform


BANDS = ("B02", "B03", "B04", "B08")
CHIP_SIZE = 256
NODATA_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 4-band 256x256 HR chips from an extracted Sentinel-2 SAFE folder."
    )
    parser.add_argument(
        "--safe_dir",
        type=Path,
        required=True,
        help="Path to an extracted .SAFE folder, not the ZIP file.",
    )
    parser.add_argument(
        "--output_scene",
        type=Path,
        default=Path("./scene_stacked.tif"),
        help="Path for the full stacked 4-band GeoTIFF (default: ./scene_stacked.tif).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split (default: 42).",
    )
    parser.add_argument(
        "--save_scene_stack",
        action="store_true",
        default=False,
        help="Write the full stacked 4-band scene image to --output_scene only when explicitly requested.",
    )
    return parser.parse_args()


def find_band_paths(safe_dir: Path) -> list[Path]:
    r10m_dirs = sorted(safe_dir.glob("GRANULE/*/IMG_DATA/R10m"))
    if not r10m_dirs:
        raise FileNotFoundError(
            f"No GRANULE/*/IMG_DATA/R10m directory found under {safe_dir}. "
            "Pass the extracted .SAFE folder with --safe_dir."
        )

    band_paths: list[Path] = []
    for band in BANDS:
        matches = sorted(
            path
            for r10m_dir in r10m_dirs
            for path in r10m_dir.glob(f"*_B{band[1:]}_10m.jp2")
        )
        if not matches:
            raise FileNotFoundError(
                f"Could not find a {band} band matching *_B{band[1:]}_10m.jp2 "
                f"under {safe_dir / 'GRANULE'}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple {band} files across the SAFE granules: "
                + ", ".join(str(path) for path in matches)
                + ". Supply a SAFE with one granule."
            )
        band_paths.append(matches[0])
    return band_paths


def read_and_stack(band_paths: list[Path]) -> tuple[np.ndarray, dict]:
    arrays: list[np.ndarray] = []
    reference_profile: dict | None = None
    reference_crs = None
    reference_transform = None
    reference_shape: tuple[int, int] | None = None

    for band_path in band_paths:
        try:
            # JP2 reading requires GDAL's JP2OpenJPEG driver. It is usually bundled
            # with rasterio's wheel; reinstall with `pip install rasterio --no-binary
            # rasterio` if the driver is unavailable on a particular system.
            with rasterio.open(band_path) as source:
                array = source.read(1)
                profile = source.profile.copy()
                crs = source.crs
                transform = source.transform
                shape = (source.height, source.width)
        except Exception as error:
            raise RuntimeError(
                f"Could not read Sentinel-2 JP2 band {band_path}. GDAL needs the "
                "JP2OpenJPEG driver. It is usually bundled with rasterio's wheel; "
                "try `pip install rasterio --no-binary rasterio` as a fallback."
            ) from error

        if reference_shape is None:
            reference_profile = profile
            reference_crs = crs
            reference_transform = transform
            reference_shape = shape
        else:
            mismatches = []
            if crs != reference_crs:
                mismatches.append("CRS")
            if transform != reference_transform:
                mismatches.append("transform")
            if shape != reference_shape:
                mismatches.append("shape")
            if mismatches:
                raise ValueError(
                    f"Band {band_path.name} does not align with the first band; "
                    f"mismatched {', '.join(mismatches)}."
                )
        arrays.append(array)

    if reference_profile is None or reference_shape is None:
        raise ValueError("No bands were read from the SAFE product.")

    stack = np.stack(arrays, axis=0)
    reference_profile.update(count=len(BANDS), dtype=stack.dtype)
    return stack, reference_profile


def derive_scene_prefix(safe_dir: Path) -> str:
    """Create a stable per-scene prefix from a Sentinel-2 SAFE folder name.

    Example SAFE folder name:
    S2A_MSIL2A_20260823T053241_N0512_R105_T43PCS_20260823T122218.SAFE

    Returns a short prefix like T43PCS_20260823.
    """
    safe_name = safe_dir.name.upper()
    safe_name = safe_name[:-5] if safe_name.endswith(".SAFE") else safe_name

    # Expected SAFE naming convention: ..._T43PCS_20260823T122218
    match = re.search(r"_(T\d{2}[A-Z]{3})_(\d{8})T\d{6}$", safe_name)
    if match:
        tile_id, date_token = match.groups()
        return f"{tile_id}_{date_token}"

    # Conservative fallback: scan for any tile token and a date token anywhere
    tile_match = re.search(r"_(T\d{2}[A-Z]{3})_", safe_name)
    date_match = re.search(r"_(\d{8})T\d{6}", safe_name)
    if tile_match and date_match:
        return f"{tile_match.group(1)}_{date_match.group(1)}"

    raise ValueError(
        f"Could not derive a chip-prefix from SAFE folder name: {safe_dir.name}. "
        "Expected a Sentinel-2 SAFE folder following the S2*_MSIL2A_YYYYMMDD..._Txxxxx_...SAFE pattern."
    )


def write_scene(stack: np.ndarray, profile: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_profile = profile.copy()
    scene_profile.update(driver="GTiff", count=stack.shape[0], dtype=stack.dtype)
    with rasterio.open(output_path, "w", **scene_profile) as destination:
        destination.write(stack)


def ensure_no_scene_prefix_collision(scene_prefix: str, train_dir: Path, val_dir: Path) -> None:
    """Raise before writing when a requested scene prefix already has files in the split folders."""
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    collisions: list[Path] = []
    for split_dir in (train_dir, val_dir):
        collisions.extend(split_dir.glob(f"{scene_prefix}_chip_*.tif"))

    if collisions:
        formatted = "\n  - ".join(str(path) for path in collisions)
        loud_message = (
            f"!!! WARNING: filename-prefix collision detected for scene prefix "
            f"{scene_prefix}. Existing chip files already present in the output dataset:\n"
            f"  - {formatted}\n"
            "Refusing to continue and overwrite any existing chip files. "
            "Please remove or rename the colliding sample files before rerunning."
        )
        warnings.warn(loud_message, RuntimeWarning, stacklevel=2)
        raise FileExistsError(
            f"Refusing to overwrite chips with colliding scene prefix '{scene_prefix}'."
        )


def extract_chips(
    stack: np.ndarray,
    profile: dict,
    train_dir: Path,
    val_dir: Path,
    seed: int,
    scene_prefix: str,
) -> tuple[int, int, int, np.ndarray]:
    height, width = stack.shape[1:]
    chips: list[tuple[int, int, np.ndarray]] = []
    skipped = 0

    for row in range(0, height - CHIP_SIZE + 1, CHIP_SIZE):
        for column in range(0, width - CHIP_SIZE + 1, CHIP_SIZE):
            chip = stack[:, row : row + CHIP_SIZE, column : column + CHIP_SIZE]
            zero_pixels = np.all(chip == 0, axis=0)
            if np.count_nonzero(zero_pixels) / zero_pixels.size > NODATA_LIMIT:
                skipped += 1
                continue
            chips.append((row, column, chip))

    rng = np.random.default_rng(seed)
    rng.shuffle(chips)
    train_count = int(round(len(chips) * 0.9))
    if len(chips) > 1:
        train_count = min(max(train_count, 1), len(chips) - 1)

    train_chips = chips[:train_count]
    val_chips = chips[train_count:]
    chip_profile = profile.copy()
    chip_profile.update(
        driver="GTiff",
        count=stack.shape[0],
        width=CHIP_SIZE,
        height=CHIP_SIZE,
        dtype=stack.dtype,
    )

    ensure_no_scene_prefix_collision(scene_prefix, train_dir, val_dir)

    for split_dir, split_chips in ((train_dir, train_chips), (val_dir, val_chips)):
        split_dir.mkdir(parents=True, exist_ok=True)
        for chip_number, (row, column, chip) in enumerate(split_chips):
            window = Window(column, row, CHIP_SIZE, CHIP_SIZE)
            chip_profile["transform"] = window_transform(window, profile["transform"])
            output_path = split_dir / f"{scene_prefix}_chip_{chip_number:04d}.tif"
            with rasterio.open(output_path, "w", **chip_profile) as destination:
                destination.write(chip)

    if chips:
        valid_values = np.concatenate(
            [chip[2].reshape(stack.shape[0], -1) for chip in chips], axis=1
        )
        average_reflectance = np.mean(valid_values, axis=1)
    else:
        average_reflectance = np.full(stack.shape[0], np.nan)
    return len(chips), skipped, len(train_chips), average_reflectance


def main() -> None:
    args = parse_args()
    safe_dir = args.safe_dir.expanduser().resolve()
    if not safe_dir.is_dir():
        raise FileNotFoundError(f"SAFE directory does not exist: {safe_dir}")
    if safe_dir.suffix.upper() != ".SAFE":
        raise ValueError(f"--safe_dir must point to an extracted .SAFE folder: {safe_dir}")

    with Env(CHECK_DISK_FREE_SPACE='FALSE'):
        scene_prefix = derive_scene_prefix(safe_dir)

        band_paths = find_band_paths(safe_dir)
        stack, profile = read_and_stack(band_paths)

        if args.save_scene_stack:
            write_scene(stack, profile, args.output_scene)

        train_dir = Path("datasets/samples/train/hr")
        val_dir = Path("datasets/samples/val/hr")
        total_chips, skipped, train_count, average_reflectance = extract_chips(
            stack, profile, train_dir, val_dir, args.seed, scene_prefix
        )

        print("SAFE training-chip extraction complete")
        print(f"Stacked scene: {args.output_scene}")
        print(f"Total chips extracted: {total_chips}")
        print(f"Chips skipped for nodata (>5% zero pixels): {skipped}")
        print(f"Train/val split: {train_count}/{total_chips - train_count}")
        print(
            "Average per-band reflectance (B02, B03, B04, B08): "
            + ", ".join(f"{value:.2f}" for value in average_reflectance)
        )


if __name__ == "__main__":
    main()
