"""Convert an extracted Sentinel-2 SAFE product into HR training chips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
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


def write_scene(stack: np.ndarray, profile: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_profile = profile.copy()
    scene_profile.update(driver="GTiff", count=stack.shape[0], dtype=stack.dtype)
    with rasterio.open(output_path, "w", **scene_profile) as destination:
        destination.write(stack)


def extract_chips(
    stack: np.ndarray,
    profile: dict,
    train_dir: Path,
    val_dir: Path,
    seed: int,
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

    for split_dir, split_chips in ((train_dir, train_chips), (val_dir, val_chips)):
        split_dir.mkdir(parents=True, exist_ok=True)
        for chip_number, (row, column, chip) in enumerate(split_chips):
            window = Window(column, row, CHIP_SIZE, CHIP_SIZE)
            chip_profile["transform"] = window_transform(window, profile["transform"])
            output_path = split_dir / f"chip_{chip_number:04d}.tif"
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

    band_paths = find_band_paths(safe_dir)
    stack, profile = read_and_stack(band_paths)
    write_scene(stack, profile, args.output_scene)

    train_dir = Path("datasets/samples/train/hr")
    val_dir = Path("datasets/samples/val/hr")
    total_chips, skipped, train_count, average_reflectance = extract_chips(
        stack, profile, train_dir, val_dir, args.seed
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
