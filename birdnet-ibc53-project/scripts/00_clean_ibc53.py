"""
Script 0: Clean IBC53 Dataset
================================
Removes species folders not specified in the PRD/Implementation Guide.
Keeps only the 30 selected species (>=10 files) + the Mystery mystery folder.
Deletes the 22 dropped species (<10 files each).

Usage:
    python scripts/00_clean_ibc53.py
    python scripts/00_clean_ibc53.py --dry_run          # Preview only, no deletions
    python scripts/00_clean_ibc53.py --ibc53_dir data/ibc53

Pipeline Position: Run ONCE after downloading and extracting the IBC53 dataset.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    IBC53_RAW_DIR, SPECIES_NAMES, SPECIES_COMMON_NAMES,
    MYSTERY_FOLDER_NAME,
)

# Folders to keep: 30 selected species + Mystery mystery
KEEP_FOLDERS = set(SPECIES_NAMES) | {MYSTERY_FOLDER_NAME}


def clean_ibc53(ibc53_dir: Path, dry_run: bool = False) -> dict:
    """
    Remove species folders not in the selected 30 + Mystery mystery.

    Args:
        ibc53_dir: Path to the raw IBC53 dataset directory.
        dry_run: If True, only print what would be deleted without deleting.

    Returns:
        dict with kept/deleted folder lists.
    """
    if not ibc53_dir.is_dir():
        print(f"[ERROR] IBC53 directory not found: {ibc53_dir}")
        print("  Download first: kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset")
        return {}

    all_folders = sorted([d for d in ibc53_dir.iterdir() if d.is_dir()])
    all_files = [f for f in ibc53_dir.iterdir() if f.is_file()]

    kept = []
    deleted = []
    not_found = []

    print("=" * 70)
    print("IBC53 DATASET CLEANUP")
    print(f"Directory: {ibc53_dir}")
    print(f"Mode:      {'DRY RUN (no deletions)' if dry_run else 'LIVE (will delete folders)'}")
    print("=" * 70)

    # Check which expected folders exist
    for folder_name in sorted(KEEP_FOLDERS):
        match = ibc53_dir / folder_name
        if match.is_dir():
            n_files = len([f for f in match.iterdir() if f.is_file()])
            common = SPECIES_COMMON_NAMES.get(folder_name, "")
            label = f"({common})" if common else "(Mystery folder)"
            kept.append(folder_name)
            print(f"  [KEEP]   {folder_name:45s} {label:40s} | {n_files:3d} files")
        else:
            # Try case-insensitive match
            matches = [d for d in all_folders if d.name.lower() == folder_name.lower()]
            if matches:
                actual = matches[0]
                n_files = len([f for f in actual.iterdir() if f.is_file()])
                common = SPECIES_COMMON_NAMES.get(folder_name, "")
                label = f"({common})" if common else "(Mystery folder)"
                kept.append(actual.name)
                print(f"  [KEEP]   {actual.name:45s} {label:40s} | {n_files:3d} files")
            else:
                not_found.append(folder_name)

    print()

    # Identify and delete folders not in the keep list
    for folder in all_folders:
        if folder.name not in KEEP_FOLDERS:
            # Case-insensitive check
            if not any(folder.name.lower() == k.lower() for k in KEEP_FOLDERS):
                n_files = len([f for f in folder.iterdir() if f.is_file()])
                deleted.append(folder.name)
                print(f"  [DELETE] {folder.name:45s} | {n_files:3d} files")
                if not dry_run:
                    shutil.rmtree(folder)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"CLEANUP SUMMARY")
    print(f"  Folders kept:      {len(kept)}")
    print(f"  Folders deleted:   {len(deleted)}")
    if not_found:
        print(f"  Expected but missing: {len(not_found)}")
        for name in not_found:
            print(f"    - {name}")
    if dry_run and deleted:
        print(f"\n  *** DRY RUN — no folders were actually deleted ***")
        print(f"  *** Run without --dry_run to delete ***")
    elif not dry_run and deleted:
        print(f"\n  {len(deleted)} folders permanently deleted.")
    elif not deleted:
        print(f"\n  Nothing to delete — dataset is already clean.")
    print(f"{'=' * 70}")

    return {"kept": kept, "deleted": deleted, "not_found": not_found}


def main():
    parser = argparse.ArgumentParser(
        description="Remove IBC53 species not in the selected 30. "
                    "Run after downloading and extracting the dataset."
    )
    parser.add_argument("--ibc53_dir", type=str, default=str(IBC53_RAW_DIR),
                        help="Path to IBC53 dataset directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview deletions without actually deleting")
    args = parser.parse_args()

    clean_ibc53(
        ibc53_dir=Path(args.ibc53_dir),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
