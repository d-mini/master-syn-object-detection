#!/usr/bin/env python3
import argparse, json, random, shutil, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

RANDOM_SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ------------------------------
# YOLO flat dataset loading
# ------------------------------

def parse_yolo_label_file(path: Path) -> list[int]:
    if not path.exists():
        return []
    cids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(parts[0])
            except Exception:
                continue
            cids.append(cid)
    return cids

def read_flat_yolo(root: Path):
    """
    Erwartet:
      root/images/*.ext
      root/labels/*.txt
      optional: root/classes.txt mit Zeilen = Klassennamen
    Gibt zurück:
      {
        "images": [ { "file": str, "labels_file": str, "classes_present": set[int], "inst_per_class": Counter } ],
        "counts_per_class": dict[int,int],
        "categories": [ {id,name} ],
        "classes_list": [str]
      }
    """
    img_dir = root / "images"
    lab_dir = root / "labels"
    if not img_dir.exists() or not lab_dir.exists():
        print(f"Erwarte YOLO Layout unter {root}/images und {root}/labels.", file=sys.stderr)
        sys.exit(1)

    # optional classes.txt
    classes_txt = root / "classes.txt"
    classes_from_file = None
    if classes_txt.exists():
        classes_from_file = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]

    images = []
    counts = defaultdict(int)
    max_cid = -1

    for p in img_dir.glob("*"):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        lab = lab_dir / (p.stem + ".txt")
        cids = parse_yolo_label_file(lab)
        for cid in cids:
            counts[cid] += 1
            if cid > max_cid:
                max_cid = cid
        images.append({
            "file": str(p),
            "labels_file": str(lab),
            "classes_present": set(cids),
            "inst_per_class": Counter(cids),
        })

    # Kategorien / Klassenliste
    if classes_from_file is not None:
        classes_list = classes_from_file
        categories = [{"id": i, "name": classes_list[i] if i < len(classes_list) else f"class_{i}"} for i in range(len(classes_list))]
    else:
        n = max(0, max_cid + 1)
        classes_list = [f"class_{i}" for i in range(n)]
        categories = [{"id": i, "name": classes_list[i] if i < len(classes_list) else f"class_{i}"} for i in range(len(classes_list))]

    return {
        "images": images,
        "counts_per_class": dict(counts),
        "categories": categories,
        "classes_list": classes_list,
        "img_dir": str(img_dir),
        "lab_dir": str(lab_dir),
    }

# ------------------------------
# Candidates aus flachem Datensatz
# ------------------------------

def build_candidates(dataset):
    candidates = []
    if not dataset or "images" not in dataset:
        return candidates

    for rec in dataset["images"]:
        classes_present = rec["classes_present"]
        inst_per_class = rec["inst_per_class"]
        if len(classes_present) == 0:
            continue

        if len(classes_present) > 1:
            group_name = "multiclass"
        else:
            only_cls = next(iter(classes_present))
            group_name = "multi_instance" if inst_per_class[only_cls] > 1 else "single"

        candidates.append({
            "group": group_name,
            "file": rec["file"],          # Pfad zur Bilddatei
            "label": rec["labels_file"],  # Pfad zur Labeldatei
            "contrib": dict(inst_per_class),  # {cid: anzahl in diesem bild}
        })
    return candidates

# ------------------------------
# Seed phase (leicht angepasst auf "file"-Key)
# ------------------------------

def seed_per_group_class_min1(categories_order, candidates, total, min_per, max_per):
    already = set()  # set[(group, file)]
    group_counts = Counter()
    selected = []

    has_pair = defaultdict(bool)
    for c in candidates:
        g = c["group"]
        for cid in categories_order:
            if c["contrib"].get(cid, 0) > 0:
                has_pair[(g, cid)] = True

    def deficit_to_min(cid):
        return max(0, min_per - total[cid])

    groups = sorted({c["group"] for c in candidates})
    for g in groups:
        group_cands = [c for c in candidates if c["group"] == g]
        for cid in categories_order:
            if not has_pair[(g, cid)]:
                continue

            covered = any(sc["group"] == g and sc["contrib"].get(cid, 0) > 0 for sc in selected)
            if covered:
                continue

            best, best_s = None, -1e9
            for c in group_cands:
                key = (c["group"], c["file"])
                if key in already or c["contrib"].get(cid, 0) <= 0:
                    continue
                # nicht über max_per schießen
                if any(total[k] + c["contrib"].get(k, 0) > max_per for k in categories_order):
                    continue

                gain_main = min(deficit_to_min(cid), c["contrib"].get(cid, 0))
                if gain_main == 0:
                    gain_main = 1
                multi_bonus = sum(
                    1 for k in categories_order
                    if k != cid and deficit_to_min(k) > 0 and c["contrib"].get(k, 0) > 0
                ) * 0.1
                s = gain_main + multi_bonus
                if s > best_s:
                    best, best_s = c, s

            if best is None:
                continue

            key = (best["group"], best["file"])
            already.add(key)
            selected.append(best)
            group_counts[g] += 1
            for k in categories_order:
                total[k] += best["contrib"].get(k, 0)

    return selected, already, group_counts, total

# ------------------------------
# Greedy (angepasst auf "file"-Key)
# ------------------------------

def greedy_global_sampling(categories_order, quota, target_per_class, tolerance, candidates,
                           start_selected=None, start_totals=None, start_group_counts=None, start_already=None,
                           seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    min_per = int(target_per_class * (1 - tolerance))
    max_per = int(target_per_class * (1 + tolerance))

    selected = list(start_selected) if start_selected else []
    total = dict(start_totals) if start_totals else {cid: 0 for cid in categories_order}
    group_counts = Counter(start_group_counts) if start_group_counts else Counter()
    already = set(start_already) if start_already else set()

    shuffled = list(candidates)
    random.shuffle(shuffled)

    if candidates:
        avg_inst_per_img = sum(sum(c["contrib"].values()) for c in candidates) / len(candidates)
    else:
        avg_inst_per_img = 1.0
    expected_total_instances = target_per_class * len(categories_order)
    expected_total_images = max(1, int(round(expected_total_instances / max(1.0, avg_inst_per_img))))

    def deficit_to_min(cid): return max(0, min_per - total[cid])
    def deficit_to_target(cid): return max(0, target_per_class - total[cid])
    def all_within_band(): return all(min_per <= total[cid] <= max_per for cid in categories_order)

    def score_candidate(c):
        if not any(deficit_to_target(cid) > 0 and c["contrib"].get(cid, 0) > 0 for cid in categories_order):
            return -1e9

        proj = {cid: total[cid] + c["contrib"].get(cid, 0) for cid in categories_order}
        if any(proj[cid] > max_per for cid in categories_order):
            return -1e9

        gain_min = sum(min(deficit_to_min(cid), c["contrib"].get(cid, 0)) for cid in categories_order)
        residual_contrib = {
            cid: max(0, c["contrib"].get(cid, 0) - min(deficit_to_min(cid), c["contrib"].get(cid, 0)))
            for cid in categories_order
        }
        gain_target = sum(min(deficit_to_target(cid), residual_contrib[cid]) for cid in categories_order)
        gain = gain_min + 0.25 * gain_target

        g = c["group"]
        desired_cnt = quota.get(g, 0.0) * expected_total_images
        gap = max(0.0, desired_cnt - group_counts[g])
        group_bonus = 0.02 * gap

        serves_min_need = any(deficit_to_min(cid) > 0 and c["contrib"].get(cid, 0) > 0 for cid in categories_order)
        if serves_min_need:
            multi_cover = sum(1 for cid in categories_order if c["contrib"].get(cid, 0) > 0 and deficit_to_min(cid) > 0)
            multi_bonus = 0.05 * multi_cover
        else:
            multi_cover = sum(1 for cid in categories_order if c["contrib"].get(cid, 0) > 0 and deficit_to_target(cid) > 0)
            multi_bonus = 0.02 * multi_cover

        return gain + group_bonus + multi_bonus

    while not all_within_band():
        best_idx, best_score = None, -1e9
        for idx, c in enumerate(shuffled):
            key = (c["group"], c["file"])
            if key in already:
                continue
            if not any(deficit_to_target(cid) > 0 and c["contrib"].get(cid, 0) > 0 for cid in categories_order):
                continue
            s = score_candidate(c)
            if s > best_score:
                best_score, best_idx = s, idx

        if best_idx is None or best_score <= 0:
            print("[INFO] Kein sinnvoller Fortschritt mehr möglich; breche ab.", file=sys.stderr)
            break

        c = shuffled[best_idx]
        key = (c["group"], c["file"])
        already.add(key)
        selected.append(c)
        group_counts[c["group"]] += 1
        for cid in categories_order:
            total[cid] += c["contrib"].get(cid, 0)

    remaining_support = Counter()
    for c in candidates:
        key = (c["group"], c["file"])
        if key in already:
            continue
        for cid in categories_order:
            if c["contrib"].get(cid, 0) > 0:
                remaining_support[cid] += c["contrib"][cid]

    debug = {
        "expected_total_images": expected_total_images,
        "remaining_support": remaining_support,
    }
    return selected, total, (min_per, max_per), group_counts, debug

# ------------------------------
# Merge & Copy (flat)
# ------------------------------

def merge_and_copy_selected_flat(dataset, selected, out_dir: Path, categories, classes_list):
    out_img_dir = out_dir / "images"
    out_lab_dir = out_dir / "labels"
    out_leftover_dir = out_dir / "leftover_images"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)
    out_leftover_dir.mkdir(parents=True, exist_ok=True)

    # write classes.txt
    if classes_list is None:
        classes_list = [c["name"] for c in categories]
    (out_dir / "classes.txt").write_text("\n".join(classes_list) + "\n", encoding="utf-8")

    merged = 0
    per_class_inst = Counter()
    group_count = Counter()

    file_set = set()

    for item in selected:
        file_set.add(item["file"])
        src_img = Path(item["file"])
        src_lab = Path(item["label"])

        # Zielnamen (Kollisionen vermeiden)
        dst_img = out_img_dir / src_img.name
        dst_lab = out_lab_dir / (dst_img.stem + ".txt")

        shutil.copy2(src_img, dst_img)
        if src_lab.exists():
            shutil.copy2(src_lab, dst_lab)
        else:
            dst_lab.write_text("", encoding="utf-8")

        # Accounting
        for cid, n in item["contrib"].items():
            per_class_inst[cid] += n
        group_count[item["group"]] += 1
        merged += 1

    # Leftover (optional)
    leftovers = set()
    for image in dataset["images"]:
        if image["file"] in file_set:
            continue
        
        leftovers.add(image["file"])

    sampled_leftover = np.random.choice(list(leftovers), size=min(len(leftovers), 100), replace=False)
    for lf in sampled_leftover:
        src_img = Path(lf)
        dst_img = out_leftover_dir / src_img.name
        shutil.copy2(src_img, dst_img)

    print(f"[OK] Bilder kopiert: {merged} → {out_img_dir}")
    print(f"[OK] Labels kopiert/erstellt → {out_lab_dir}")
    print(f"[OK] Klassen-Datei → {(out_dir / 'classes.txt')}")
    print("\nBildanzahl pro (abgeleiteter) Gruppe:")
    for g, n in group_count.items():
        print(f"  {g}: {n}")

    print("\nInstanzen pro Klasse im Merge:")
    for c in categories:
        cid = c["id"]
        print(f"  {c['name']}: {per_class_inst.get(cid, 0)}")

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("mixed"), help="Wurzelordner mit images/ und labels/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target_per_class", type=int, default=175)
    ap.add_argument("--tolerance", type=float, default=0.1, help="±Toleranz, z.B. 0.2 → 80..120")
    ap.add_argument("--q_single", type=float, default=0.2)
    ap.add_argument("--q_multiclass", type=float, default=0.4)
    ap.add_argument("--q_multi_instance", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--enforce_per_group_class_min1", action="store_true",
                    help="Seed-Phase: pro (Gruppe, Klasse) mindestens 1 Bild (falls möglich)")
    args = ap.parse_args()

    random.seed(args.seed)

    dataset = read_flat_yolo(args.root)
    categories = dataset["categories"]
    classes_list = dataset["classes_list"]
    categories_order = [c["id"] for c in categories]
    cat_name = {c["id"]: c["name"] for c in categories}

    # Quoten
    quota = {"single": args.q_single, "multiclass": args.q_multiclass, "multi_instance": args.q_multi_instance}

    # Kandidaten bauen
    candidates = build_candidates(dataset)
    if not candidates:
        print("Keine Kandidaten gefunden (evtl. fehlen Labels?).", file=sys.stderr)
        sys.exit(1)

    # Zielband
    min_per = int(args.target_per_class * (1 - args.tolerance))
    max_per = int(args.target_per_class * (1 + args.tolerance))

    # Startzustand (Seed optional)
    start_selected, start_already, start_group_counts = [], set(), Counter()
    start_totals = {cid: 0 for cid in categories_order}

    if args.enforce_per_group_class_min1:
        seed_selected, seed_already, seed_group_counts, start_totals = seed_per_group_class_min1(
            categories_order, candidates, start_totals, min_per, max_per
        )
        start_selected = seed_selected
        start_already = seed_already
        start_group_counts = seed_group_counts
        print(f"[INFO] Seed-Auswahl (per group/class): {len(seed_selected)} Bilder")
    else:
        print("[INFO] Seed-Phase übersprungen (Flag nicht gesetzt).")

    # Greedy-Sampling
    selected, totals, band, group_counts, debug = greedy_global_sampling(
        categories_order=categories_order,
        quota=quota,
        target_per_class=args.target_per_class,
        tolerance=args.tolerance,
        candidates=candidates,
        start_selected=start_selected,
        start_totals=start_totals,
        start_group_counts=start_group_counts,
        start_already=start_already,
        seed=args.seed
    )
    min_per, max_per = band

    print("\nZielband je Klasse:", f"{min_per}..{max_per}")
    print("Erreichte Totale:")
    for cid in categories_order:
        flag = ""
        if totals[cid] < min_per:
            flag = "  (UNTER min)"
        elif totals[cid] > max_per:
            flag = "  (ÜBER max)"
        print(f"  {cat_name[cid]}: {totals[cid]}{flag}")

    if any(totals[cid] < min_per for cid in categories_order):
        print("\n[INFO] Verbleibendes Instanzpotenzial (über noch nicht gewählte Bilder):")
        for cid in categories_order:
            if totals[cid] < min_per:
                print(f"  {cat_name[cid]}: restliche verfügbare Annotations ≈ {debug['remaining_support'][cid]}")

    # Merge + Copy
    args.out.mkdir(parents=True, exist_ok=True)
    merge_and_copy_selected_flat(dataset, selected, args.out, categories, classes_list)

if __name__ == "__main__":
    main()
