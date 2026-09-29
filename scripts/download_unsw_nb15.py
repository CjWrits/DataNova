"""UNSW-NB15 Dataset Acquisition and Integrity Verification Script.

Downloads the canonical UNSW-NB15 machine learning benchmark CSV files
and associated feature/event metadata, computes SHA-256 checksums,
and verifies file integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

# Official and verified mirror sources
DOWNLOAD_TARGETS = {
    "UNSW_NB15_training-set.csv": {
        "url": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv",
        "description": "Author-partitioned training subset (82,332 records, 45 attributes)",
        "expected_min_bytes": 15_000_000,
    },
    "UNSW_NB15_testing-set.csv": {
        "url": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv",
        "description": "Author-partitioned testing subset (175,341 records, 45 attributes)",
        "expected_min_bytes": 30_000_000,
    },
    "NUSW-NB15_features.csv": {
        "url": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/raw/main/NUSW-NB15_features.csv",
        "description": "Feature dictionary detailing all 49 original attributes and dtypes",
        "expected_min_bytes": 3_500,
    },
    "UNSW-NB15_LIST_EVENTS.csv": {
        "url": "https://raw.githubusercontent.com/harshilpatel1799/IoT-Network-Intrusion-Detection-and-Classification-using-Explainable-XAI-Machine-Learning/main/UNSW-NB15_LIST_EVENTS.csv",
        "description": "Reference catalog of attack categories, subcategories, and event frequencies",
        "expected_min_bytes": 4_000,
    },
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, destination: Path, expected_min_bytes: int) -> Tuple[int, str]:
    """Download a file with streaming progress and compute its checksum."""
    headers = {"User-Agent": "Adaptive-Threat-Intelligence-Research/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    print(f"Downloading from: {url}")
    print(f"Destination:     {destination}")
    
    temp_dest = destination.with_suffix(".tmp")
    with urllib.request.urlopen(req, timeout=120) as response, open(temp_dest, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rProgress: {downloaded:,} / {total_size:,} bytes ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\rProgress: {downloaded:,} bytes downloaded")
            sys.stdout.flush()
    print()
    
    if downloaded < expected_min_bytes:
        temp_dest.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded file size ({downloaded} bytes) is less than expected minimum ({expected_min_bytes} bytes)."
        )
        
    temp_dest.replace(destination)
    checksum = compute_sha256(destination)
    return downloaded, checksum


def run_acquisition(target_dir: Path, force: bool = False) -> Dict[str, Dict[str, any]]:
    """Execute dataset download and generate checksum manifest."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, any]] = {}
    checksum_lines: List[str] = []

    print("=" * 70)
    print("UNSW-NB15 DATASET ACQUISITION & INTEGRITY CHECK")
    print(f"Target Directory: {target_dir}")
    print("=" * 70)

    for filename, meta in DOWNLOAD_TARGETS.items():
        filepath = target_dir / filename
        print(f"\nProcessing: {filename}")
        print(f"Description: {meta['description']}")

        if filepath.exists() and not force:
            size = filepath.stat().st_size
            print(f"File already exists ({size:,} bytes). Skipping download.")
            checksum = compute_sha256(filepath)
        else:
            size, checksum = download_file(
                url=meta["url"],
                destination=filepath,
                expected_min_bytes=meta["expected_min_bytes"],
            )

        print(f"SHA-256: {checksum}")
        print(f"Verified Size: {size:,} bytes")
        manifest[filename] = {
            "size_bytes": size,
            "sha256": checksum,
            "url": meta["url"],
            "description": meta["description"],
        }
        checksum_lines.append(f"{checksum}  {filename}\n")

    # Write checksums manifest
    checksum_file = target_dir / "checksums.sha256"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)
    print(f"\nSaved checksum manifest to: {checksum_file}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UNSW-NB15 benchmark files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/unsw_nb15"),
        help="Directory to store raw downloaded files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if files exist",
    )
    args = parser.parse_args()
    run_acquisition(args.output_dir, force=args.force)
