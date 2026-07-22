from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORY_FILE = BASE_DIR / "data" / "raw" / "FoodSeg103" / "category_id.txt"
YAML_OUT = BASE_DIR / "food_config.yaml"

names_dict = {}

if CATEGORY_FILE.exists():
    with open(CATEGORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                cat_id = int(parts[0]) - 1  # 0-indexed for YOLO
                cat_name = parts[1].lower().strip()
                names_dict[cat_id] = cat_name

config_data = {
    "path": str(BASE_DIR / "data" / "yolo_dataset"),
    "train": "images/train",
    "val": "images/val",
    "nc": len(names_dict),
    "names": names_dict
}

with open(YAML_OUT, "w") as f:
    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

print(f"Successfully generated food_config.yaml with {len(names_dict)} classes!")