import json
import shutil
from pathlib import Path
from collections import defaultdict

# === Konfiguration ===
splits = ["train", "val", "test"]
target_classes = {
    46: "Wrench",
    37: "Screwdriver",
    14: "Hammer",
    26: "Pliers",
}

# === Pfade === !!! Anpassen falls nötig !!!

input_json_template = "{}v4.json"
input_image_template = "{}v4"
output_dir = Path("filtered/one_class_multi_inst")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
output_json = output_dir / "one_class_multi_inst.json"

# === Neue ID‑Mappings ===
old_to_new_cat_id = {old_id: new_id+1 for new_id, old_id in enumerate(sorted(target_classes))}
new_categories = [
    {"supercategory": "none", "id": new_id, "name": target_classes[old_id]}
    for old_id, new_id in old_to_new_cat_id.items()
]

# === Neue COCO‑Struktur ===
new_coco = {
    "images": [],
    "annotations": [],
    "categories": new_categories
}

new_img_id = 0
new_ann_id = 0

for split in splits:
    print(f"🔄 Verarbeite {split}...")

    # 1) Lade Original-JSON
    with open(input_json_template.format(split), 'r') as f:
        coco = json.load(f)

    # 2) Filtere nur reale Bilder (keine synthetischen)
    real_images = {
        img["id"]: img
        for img in coco["images"]
        if not img["file_name"].startswith("Image")
    }

    # 3) Zähle pro Bild und Kategorie, wie viele Instanzen existieren
    ann_count = defaultdict(lambda: defaultdict(int))
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in real_images and cat_id in old_to_new_cat_id:
            ann_count[img_id][cat_id] += 1

    # 4) Wähle Bilder, bei denen genau 1 Kategorie erscheint und ≥2 Instanzen davon !!! anpassen je nachdem, was gewünscht ist !!!
    eligible_img_ids = set()
    for img_id, cats in ann_count.items():
        if len(cats) == 1 and next(iter(cats.values())) >= 2:
            eligible_img_ids.add(img_id)
    print(f"   → Gefundene Bilder mit genau 1 Klasse und ≥2 Instanzen: {len(eligible_img_ids)}")

    # 5) Kopiere Bilder und Annotations
    img_id_map = {}
    for ann in coco["annotations"]:
        orig_img_id = ann["image_id"]
        orig_cat_id = ann["category_id"]

        # Nur relevante Annotations in erlaubten Bildern
        if orig_img_id not in eligible_img_ids or orig_cat_id not in old_to_new_cat_id:
            continue

        # Bild einmalig kopieren und neue ID zuweisen
        if orig_img_id not in img_id_map:
            img_info = real_images[orig_img_id]
            src = Path(input_image_template.format(split)) / img_info["file_name"]
            dst = output_dir / "images" / img_info["file_name"]
            if src.exists():
                shutil.copy(src, dst)
            new_img = img_info.copy()
            new_img["id"] = new_img_id
            new_coco["images"].append(new_img)
            img_id_map[orig_img_id] = new_img_id
            new_img_id += 1

        # Annotation übernehmen und neue IDs setzen
        new_ann = ann.copy()
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = img_id_map[orig_img_id]
        new_ann["category_id"] = old_to_new_cat_id[orig_cat_id]
        del new_ann["segmentation"]  # Entferne Segmentation, da nicht benötigt
        del new_ann["ignore"]

        new_coco["annotations"].append(new_ann)
        new_ann_id += 1

# === Speichern ===
with open(output_json, 'w') as f:
    json.dump(new_coco, f, indent=2)

print(f"\n✅ Gefiltert und gespeichert unter: {output_json}")
print(f"📸 Bilder insgesamt: {len(new_coco['images'])}")
print(f"🏷️ Annotations insgesamt: {len(new_coco['annotations'])}")
print(f"🗂️ Klassen: {len(new_coco['categories'])}")
