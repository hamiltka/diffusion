import os
import csv
import numpy as np
from geodiffusion.dataloader.dataset import VectorRoadDataset

# User parameters
DATA_ROOT = "/home/hamiltka/GEODiffusion/data_distribution_test"
EXCLUDE_CSV = "/home/hamiltka/GEODiffusion/excluded_crops.csv"
SPLITS = ["train", "val"]
SPARSE_THRESH = 0.01  # e.g. <1% road pixels
DENSE_THRESH = 0.20   # e.g. >20% road pixels


CSV_HEADER = [
    "crop_id","split","source_tile","has_nir","nodata_flagged","nodata_fraction",
    "has_valid_roads","road_density","density_too_sparse","density_too_dense",
    "in_blocklist","exclusion_reasons"
]

def compute_density(sample):
    mask = sample.get("road_mask")
    if mask is None:
        return None
    return float(np.sum(mask > 0)) / mask.size

def get_row(sample, split, density, sparse, dense):
    crop_id = sample.get("crop_id", sample.get("index", ""))
    source_tile = sample.get("source_tile", "")
    has_nir = sample.get("has_nir", False)
    nodata_flagged = sample.get("nodata_flagged", False)
    nodata_fraction = sample.get("nodata_fraction", 0.0)
    has_valid_roads = sample.get("has_valid_roads", True)
    in_blocklist = sample.get("in_blocklist", False)
    reasons = []
    if sparse:
        reasons.append("density_sparse")
    if dense:
        reasons.append("density_dense")
    exclusion_reasons = "|".join(reasons)
    return [
        crop_id, split, source_tile, has_nir, nodata_flagged, nodata_fraction,
        has_valid_roads, density, sparse, dense, in_blocklist, exclusion_reasons
    ]

def main():
    # Read existing crop_ids to avoid duplicates
    existing = set()
    if os.path.exists(EXCLUDE_CSV):
        with open(EXCLUDE_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["crop_id"])
    new_rows = []
    for split in SPLITS:
        ds = VectorRoadDataset(data_root=DATA_ROOT, split=split)
        for i in range(len(ds)):
            sample = ds[i]
            if sample is None or sample.get("index", -1) == -1:
                continue
            density = compute_density(sample)
            if density is None:
                continue
            sparse = density < SPARSE_THRESH
            dense = density > DENSE_THRESH
            if sparse or dense:
                crop_id = sample.get("crop_id", sample.get("index", ""))
                if crop_id in existing:
                    continue
                row = get_row(sample, split, density, sparse, dense)
                new_rows.append(row)
                existing.add(crop_id)
    if new_rows:
        file_exists = os.path.exists(EXCLUDE_CSV)
        with open(EXCLUDE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADER)
            for row in new_rows:
                writer.writerow(row)
        print(f"Added {len(new_rows)} samples to {EXCLUDE_CSV}")
    else:
        print("No new samples to exclude.")

if __name__ == "__main__":
    main()
