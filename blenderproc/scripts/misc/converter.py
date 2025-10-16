import argparse
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm as TQDM

parser = argparse.ArgumentParser()
parser.add_argument('--annotations', required=True,
                    help='Path to COCO annotations file')
args = parser.parse_args()

def convert_coco_to_yolo(
    annotations
):
    if not Path(annotations).exists():
        raise FileNotFoundError(f"annotations file {annotations} not found!")
    
    fn = Path(annotations).parent / "annotations"
    fn.mkdir(parents=True, exist_ok=True)

    with open(annotations, encoding="utf-8") as f:
        data = json.load(f)

    # Create image dict
    images = {f"{x['id']:d}": x for x in data["images"]}
    # Create image-annotations dict
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    # Write annotations file
    for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {annotations}"):
        img = images[f"{img_id:d}"]
        h, w = img["height"], img["width"]
        f = img["file_name"]
        if '/' in f:
            f = f.split('/')[-1]

        bboxes = []
        for ann in anns:
            if ann.get("iscrowd", False):
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = ann["category_id"] - 1  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)

        # Write
        with open((fn / f).with_suffix(".txt"), "w", encoding="utf-8") as file:
            for i in range(len(bboxes)):
                line = (
                    *(bboxes[i]),
                )  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")

            categories = data["categories"]
    categories = sorted(categories, key=lambda x: x["id"])

    print(f"COCO data converted successfully.\nResults saved to {fn}")

convert_coco_to_yolo(annotations=args.annotations)