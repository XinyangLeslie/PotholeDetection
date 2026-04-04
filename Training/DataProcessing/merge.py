import os
import re
import csv
import json
import shutil
import hashlib
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

from PIL import Image
import imagehash
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def md5_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def read_yolo_label_lines(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    return [line.strip() for line in txt.splitlines() if line.strip()]


def filter_yolo_classes(lines: List[str], keep_classes: Set[int]) -> List[str]:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except:
            continue
        if cls in keep_classes:
            # force cls id mapping: keep pothole as 0
            # If keep_classes = {0}, then cls remains 0.
            parts[0] = "0"
            out.append(" ".join(parts[:5]))  # bbox only
    return out


def valid_yolo_bbox_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 5:
        return False
    try:
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
    except:
        return False
    if cls != 0:
        return False
    # allow tiny rounding errors
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False
    return True


def compute_phash(p: Path) -> imagehash.ImageHash:
    # convert to consistent mode
    with Image.open(p) as im:
        im = im.convert("RGB")
        return imagehash.phash(im)


@dataclass
class Item:
    image: Path
    label: Path
    prefix: str
    split: str  # train/valid/test


# ----------------------------
# Step 1: Collect to staging
# ----------------------------
def collect_to_staging(
    root: Path,
    datasets: List[Tuple[str, str]],
    staging: Path,
    dry_run: bool = True
) -> Path:
    """
    datasets: list of (dataset_folder_name, prefix)
    """
    staging_images = staging / "images"
    staging_labels = staging / "labels"
    ensure_dir(staging_images)
    ensure_dir(staging_labels)

    manifest_path = staging / "manifest.csv"
    rows = []

    def find_label_for_image(img_path: Path, split: str) -> Optional[Path]:
        # Expected: .../<split>/images/xxx.jpg  -> .../<split>/labels/xxx.txt
        # user structure: train/images and train/labels
        # Replace "/images/" with "/labels/" and suffix -> .txt
        parts = list(img_path.parts)
        try:
            idx = parts.index("images")
        except ValueError:
            return None
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return label_path

    for ds_name, prefix in datasets:
        ds_root = root / ds_name
        if not ds_root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {ds_root}")

        for split in ["train", "valid", "test"]:
            img_dir = ds_root / split / "images"
            if not img_dir.exists():
                continue

            img_files = [p for p in img_dir.rglob("*") if p.is_file() and is_image_file(p)]
            for img_path in tqdm(img_files, desc=f"Collect {ds_name}/{split}", leave=False):
                label_path = find_label_for_image(img_path, split)
                if label_path is None:
                    continue

                # build new name with prefix to avoid collisions
                new_stem = f"{prefix}_{img_path.stem}"
                new_img_name = new_stem + img_path.suffix.lower()
                new_lbl_name = new_stem + ".txt"

                dst_img = staging_images / new_img_name
                dst_lbl = staging_labels / new_lbl_name

                # Read and normalize labels
                lines = read_yolo_label_lines(label_path)

                # dataset_2 has two classes; keep only class 0
                # For all datasets we enforce class 0 only
                keep = {0}
                filtered = filter_yolo_classes(lines, keep)

                # validate bbox lines (optional strict)
                filtered = [ln for ln in filtered if valid_yolo_bbox_line(ln)]

                rows.append({
                    "dataset": ds_name,
                    "prefix": prefix,
                    "orig_split": split,
                    "src_image": str(img_path),
                    "src_label": str(label_path),
                    "staging_image": str(dst_img),
                    "staging_label": str(dst_lbl),
                    "num_boxes": len(filtered),
                })

                if dry_run:
                    continue

                safe_copy(img_path, dst_img)
                ensure_dir(dst_lbl.parent)
                # write label (can be empty: negative sample)
                dst_lbl.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")

    # write manifest
    if not dry_run:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)

    return manifest_path


# ----------------------------
# Step 2: Exact dedup (MD5)
# ----------------------------
def md5_dedup_inplace(staging: Path, dry_run: bool = True) -> Dict:
    images_dir = staging / "images"
    labels_dir = staging / "labels"
    img_files = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])

    md5_map: Dict[str, List[Path]] = {}
    for p in tqdm(img_files, desc="MD5 hashing"):
        h = md5_file(p)
        md5_map.setdefault(h, []).append(p)

    duplicates = {h: ps for h, ps in md5_map.items() if len(ps) > 1}

    removed = []
    for h, ps in duplicates.items():
        ps_sorted = sorted(ps, key=lambda x: x.name)  # keep the first by name
        keep = ps_sorted[0]
        for dup in ps_sorted[1:]:
            # remove dup image and its label
            lbl = labels_dir / (dup.stem + ".txt")
            removed.append({"keep": str(keep), "remove_image": str(dup), "remove_label": str(lbl)})
            if dry_run:
                continue
            dup.unlink(missing_ok=True)
            lbl.unlink(missing_ok=True)

    report = {
        "total_images": len(img_files),
        "duplicate_groups": len(duplicates),
        "removed_count": len(removed),
        "removed_samples": removed[:20],
    }
    if not dry_run:
        (staging / "md5_dedup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


# ----------------------------
# Step 3: Near-dup clustering (pHash) + Group split
# ----------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def cluster_by_phash(staging: Path, max_hamming: int = 6) -> Tuple[List[Path], List[List[int]]]:
    images_dir = staging / "images"
    img_files = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])

    # compute hashes
    hashes = []
    for p in tqdm(img_files, desc="pHash compute"):
        try:
            hashes.append(compute_phash(p))
        except Exception:
            # unreadable image -> treat as unique, but keep hash None
            hashes.append(None)

    n = len(img_files)
    dsu = DSU(n)

    # O(N^2) can be heavy if many images; okay for ~1-5k.
    # If you have >10k images, we would optimize with bucketing.
    for i in tqdm(range(n), desc="pHash clustering"):
        hi = hashes[i]
        if hi is None:
            continue
        for j in range(i + 1, n):
            hj = hashes[j]
            if hj is None:
                continue
            if (hi - hj) <= max_hamming:
                dsu.union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        clusters.setdefault(r, []).append(i)

    cluster_list = sorted(clusters.values(), key=len, reverse=True)
    return img_files, cluster_list


def pick_representative(
    img_files: List[Path],
    cluster: List[int],
    labels_dir: Path
) -> int:
    """
    Pick a representative image index for a cluster.
    Heuristic: prefer image with MORE valid pothole boxes; tie-break by filename.
    """
    best = None
    best_key = None
    for idx in cluster:
        img = img_files[idx]
        lbl = labels_dir / (img.stem + ".txt")
        lines = read_yolo_label_lines(lbl)
        valid = [ln for ln in lines if valid_yolo_bbox_line(ln)]
        key = (len(valid), img.name)  # more boxes better
        if best is None or key > best_key:
            best = idx
            best_key = key
    return best if best is not None else cluster[0]


def export_group_split(
    staging: Path,
    out_dir: Path,
    clusters: List[List[int]],
    img_files: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    keep_one_per_cluster: bool = True,
    dry_run: bool = True
) -> Dict:
    """
    Group split by clusters to avoid leakage.
    Optionally keep one representative per cluster (recommended for near-dup removal).
    """
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1

    test_ratio = 1 - train_ratio - val_ratio

    labels_dir = staging / "labels"
    images_dir = staging / "images"

    # Decide which image indices to keep
    kept_indices: List[int] = []
    cluster_rep: List[int] = []

    for c in clusters:
        if keep_one_per_cluster and len(c) > 1:
            rep = pick_representative(img_files, c, labels_dir)
            kept_indices.append(rep)
            cluster_rep.append(rep)
        else:
            # keep all
            kept_indices.extend(c)

    # Now assign clusters to splits (cluster-level)
    rng = random.Random(seed)
    cluster_ids = list(range(len(clusters)))
    rng.shuffle(cluster_ids)

    n_clusters = len(clusters)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * val_ratio)
    train_cids = set(cluster_ids[:n_train])
    val_cids = set(cluster_ids[n_train:n_train + n_val])
    test_cids = set(cluster_ids[n_train + n_val:])

    # Build mapping image index -> split
    idx_to_split: Dict[int, str] = {}
    for cid, c in enumerate(clusters):
        split = "train" if cid in train_cids else ("val" if cid in val_cids else "test")
        if keep_one_per_cluster and len(c) > 1:
            rep = pick_representative(img_files, c, labels_dir)
            idx_to_split[rep] = split
        else:
            for idx in c:
                idx_to_split[idx] = split

    # Create out dirs
    for sp in ["train", "val", "test"]:
        ensure_dir(out_dir / "images" / sp)
        ensure_dir(out_dir / "labels" / sp)

    exported = 0
    for idx in tqdm(sorted(kept_indices), desc="Export merged dataset"):
        img = img_files[idx]
        sp = idx_to_split.get(idx, "train")
        src_img = images_dir / img.name
        src_lbl = labels_dir / (img.stem + ".txt")

        dst_img = out_dir / "images" / sp / img.name
        dst_lbl = out_dir / "labels" / sp / (img.stem + ".txt")

        exported += 1
        if dry_run:
            continue
        safe_copy(src_img, dst_img)
        # label might be missing; write empty if not exist
        ensure_dir(dst_lbl.parent)
        if src_lbl.exists():
            safe_copy(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

    # write data.yaml
    yaml_text = "\n".join([
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 1",
        "names: ['pothole']",
        ""
    ])
    if not dry_run:
        (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
        (out_dir / "split_report.json").write_text(json.dumps({
            "total_clusters": n_clusters,
            "train_clusters": len(train_cids),
            "val_clusters": len(val_cids),
            "test_clusters": len(test_cids),
            "exported_images": exported,
            "keep_one_per_cluster": keep_one_per_cluster
        }, indent=2), encoding="utf-8")

    return {
        "total_clusters": n_clusters,
        "train_clusters": len(train_cids),
        "val_clusters": len(val_cids),
        "test_clusters": len(test_cids),
        "exported_images": exported,
        "keep_one_per_cluster": keep_one_per_cluster
    }


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to folder containing dataset_1, dataset_2, ...")
    ap.add_argument("--staging", type=str, default="staging", help="Staging output folder")
    ap.add_argument("--out", type=str, default="merged_pothole", help="Final merged dataset output folder")

    ap.add_argument("--phash_hamming", type=int, default=6, help="pHash max Hamming distance for near-dup clustering")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--keep_one_per_cluster", action="store_true", help="Keep only one representative image per near-dup cluster")
    ap.add_argument("--dry-run", action="store_true", help="Do not copy/delete files, only print reports")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    staging = Path(args.staging).resolve()
    out_dir = Path(args.out).resolve()

    datasets = [
        ("dataset_1", "d1"),
        ("dataset_2", "d2"),
        ("dataset_3", "d3"),
        ("dataset_pothole", "dp"),
    ]

    print(f"[Step 1] Collect to staging: {staging}")
    if not args.dry_run:
        if staging.exists():
            shutil.rmtree(staging)
        ensure_dir(staging)
    collect_to_staging(root, datasets, staging, dry_run=args.dry_run)
    print("  Done.")

    print(f"[Step 2] MD5 exact dedup in staging")
    md5_report = md5_dedup_inplace(staging, dry_run=args.dry_run)
    print(json.dumps(md5_report, indent=2))

    print(f"[Step 3] pHash clustering (near-dup) with max_hamming={args.phash_hamming}")
    img_files, clusters = cluster_by_phash(staging, max_hamming=args.phash_hamming)
    print(f"  Total images after MD5 step (dry-run may differ): {len(img_files)}")
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Largest clusters sizes (top 10): {[len(c) for c in clusters[:10]]}")

    print(f"[Step 4] Export merged dataset to: {out_dir}")
    if not args.dry_run:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir)
    split_report = export_group_split(
        staging=staging,
        out_dir=out_dir,
        clusters=clusters,
        img_files=img_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        keep_one_per_cluster=args.keep_one_per_cluster,
        dry_run=args.dry_run
    )
    print(json.dumps(split_report, indent=2))
    print("All done.")


if __name__ == "__main__":
    main()

    # python merge.py --root ../dataset --dry-run