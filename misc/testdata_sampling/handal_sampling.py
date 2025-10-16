import json
import random
from pathlib import Path
import shutil
from collections import defaultdict, Counter
from PIL import Image
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, default=Path("detections/all_detections.json"))
    p.add_argument("--src", type=Path, default=Path("handal_test_all"))
    p.add_argument("--dst", type=Path, default=Path("balanced/images"))
    p.add_argument("--coco-out", type=Path, default=Path("balanced/coco_annotations.json"))
    p.add_argument("--images-per-scene", type=int, default=3)
    p.add_argument("--min-instances", type=int, default=50)
    p.add_argument("--max-instances", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

SCENE_TO_CLASSES = {
    12: [1,5,8],
    13: [5],
    14: [3,8],
    15: [4,6],
    16: [2],
    18: [7,9],
    19: [1],
    21: [9],
    22: [2,3,8],
    23: [5],
    24: [1,4,7],
    25: [9],
    26: [3,8],
    28: [5],
    29: [1,4,7],
    30: [1,3,8],
    31: [9],
    32: [5],
    34: [2,6],
    36: [4,7],
    37: [1],
    39: [8,9],
    40: [2],
    41: [3,4],
    42: [6,7],
    43: [5],
    44: [4,5,8],
    47: [9],
    48: [1,2],
    49: [3,7],
    50: [9],
    51: [2,5],
    52: [3],
    53: [4],
    54: [6,7],
    55: [1,8],
    56: [4],
    57: [1,3,5],
    58: [6],
    59: [7],
    62: [2,3,5],
    63: [4],
    65: [1,6,7],
    66: [8,9],
    69: [7,9],
    70: [3,4,6],
}

ID_TO_LABEL = {i: f"hammer_model_{i}" for i in range(1, 10)}
CLASS_IDS = list(ID_TO_LABEL.keys())

def count_class_instances(scene_images, image_class_counts) -> Counter:
    total = Counter()
    for p in scene_images:
        total.update(image_class_counts[p])
        
    return Counter({cid: total.get(cid, 0) for cid in CLASS_IDS})

def overflow_total(counts: Counter, max_i: int) -> int:
    return sum(max(0, counts[cid] - max_i) for cid in CLASS_IDS)

def score_add(current: Counter, add: Counter, min_i: int, max_i: int) -> tuple:
    new = current + add
    overflow = overflow_total(new, max_i) # wie weit über max über alle Klassen hinweg
    benefit = sum(min(add[c], max(0, min_i - current[c])) for c in add) # wie viele der hinzugefügten Instanzen helfen, min zu erreichen
    total_add = sum(add.values())
    return (overflow, -benefit, -total_add, random.random() * 1e-9)

def main():
    args = parse_args()
    random.seed(args.seed)

    with args.detections.open("r") as f:
        detections = json.load(f)

    scene_to_images    = defaultdict(set)     # (szene) -> Bild-IDs
    image_to_classes   = defaultdict(set)     # (szene,bild) -> set(Klassen)
    image_class_counts = defaultdict(Counter) # (szene,bild) -> Counter{klasse: anzahl}

    for det in detections:
        sid, iid, cid = det["scene_id"], det["image_id"], det["category_id"]
        if sid in SCENE_TO_CLASSES and cid in ID_TO_LABEL:
            scene_to_images[sid].add(iid)

    for sid, image_ids in scene_to_images.items():
        classes = set(SCENE_TO_CLASSES[sid])
        for iid in image_ids:
            image_to_classes[(sid, iid)] = classes
            for c in classes:
                image_class_counts[(sid, iid)][c] = 1

    # 1. Auswahl: zufällig je Szene
    selected = set()
    for sid, image_ids in scene_to_images.items():
        ids = list(image_ids)
        chosen = ids if len(ids) <= args.images_per_scene else random.sample(ids, args.images_per_scene)
        selected.update((sid, iid) for iid in chosen)

    inst = count_class_instances(selected, image_class_counts)

    # 2. Auffüllen (greedy) bis alle ≥ min_instances
    while any(inst[c] < args.min_instances for c in CLASS_IDS):
        need = {c for c in CLASS_IDS if inst[c] < args.min_instances}
        candidates = [image for image, classes in image_to_classes.items()
                      if image not in selected and any(c in need for c in classes)]
        
        if not candidates:
            break

        # bestes Bild bzgl. Overflow/Benefit
        best = min(candidates, key=lambda p: score_add(inst, image_class_counts[p], args.min_instances, args.max_instances))

        s = score_add(inst, image_class_counts[best], args.min_instances, args.max_instances)

        # Abbruch, wenn Hinzufügen nichts bringt
        if -s[1] <= 0:
            break

        selected.add(best)
        inst += image_class_counts[best]

    # Bilder kopieren
    args.dst.mkdir(parents=True, exist_ok=True)

    images, filename_to_id = [], {}
    img_id = 1

    for sid, iid in sorted(selected):
        s_str, i_str = f"{sid:06d}", f"{iid:06d}"
        src = args.src / s_str / "rgb" / f"{i_str}.jpg"
        dst_name = f"{s_str}_{i_str}.jpg"
        dst = args.dst / dst_name

        shutil.copy(src, dst)
        try:
            with Image.open(dst) as im:
                w, h = im.size
        except Exception as e:
            print(f"Fehler beim Lesen von {dst}: {e}")
            continue

        images.append({"id": img_id, "file_name": dst_name, "width": w, "height": h})
        filename_to_id[dst_name] = img_id
        img_id += 1

    # 100 zufällige Bilder aus den Resten auswählen für Domain Adaptation
    leftovers_dir = args.dst / "leftovers"
    leftovers_dir.mkdir(parents=True, exist_ok=True)
    remaining = list(image_to_classes.keys() - selected)
    print(f"{len(remaining)} verbleibende Bilder, wovon 100 kopiert werden.")
    sampled = random.sample(remaining, min(100, len(remaining)))
    for sid, iid in sampled:
        s_str, i_str = f"{sid:06d}", f"{iid:06d}"
        src = args.src / s_str / "rgb" / f"{i_str}.jpg"
        dst_name = f"{s_str}_{i_str}.jpg"
        dst = args.dst / "leftovers" / dst_name

        shutil.copy(src, dst)

    # COCO-Annotationen
    annotations, ann_id = [], 1
    images_per_class = defaultdict(set)

    for det in detections:
        sid, iid, cid = det["scene_id"], det["image_id"], det["category_id"]
        if (sid, iid) not in selected or cid not in ID_TO_LABEL:
            continue

        fname = f"{sid:06d}_{iid:06d}.jpg"
        if fname not in filename_to_id:
            continue

        coco_img_id = filename_to_id[fname]
        x, y, w, h = det["bbox"]
        annotations.append({
            "id": ann_id,
            "image_id": coco_img_id,
            "category_id": cid,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1
        images_per_class[cid].add(coco_img_id)

    categories = [{"id": i, "name": name} for i, name in ID_TO_LABEL.items()]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    with args.coco-out.open("w") as f:
        json.dump(coco, f, indent=2)

    print(f"{len(images)} Bilder, {len(annotations)} Annotations, {len(categories)} Kategorien")

    final_counts = count_class_instances(selected, image_class_counts)
    print("\nVerteilung pro Klasse:")
    for cid in sorted(CLASS_IDS):
        label = ID_TO_LABEL[cid]
        n_inst = final_counts.get(cid, 0)
        n_imgs = len(images_per_class.get(cid, set()))
        status = "OK" if (args.min_instances <= n_inst <= args.max_instances) else ("LOW" if n_inst < args.min_instances else "HIGH")
        print(f"  - {label:>16s}: {n_inst:4d} Instanzen in {n_imgs:3d} Bildern [{status}]")

if __name__ == "__main__":
    main()
