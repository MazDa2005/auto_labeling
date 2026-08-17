"""Исправляет пути к картинкам в JSON-файлах аннотаций"""
import json
from pathlib import Path

ann_dir = Path("projects/test/ann")
frames_dir = Path("projects/test/frames")

for json_path in ann_dir.glob("*.json"):
    if json_path.name.startswith("_"):
        continue
    
    with open(json_path) as f:
        data = json.load(f)
    
    image_field = data.get("image", "")
    
    # Если путь уже полный и существует — пропускаем
    if Path(image_field).exists():
        continue
    
    # Ищем картинку в frames/
    stem = Path(image_field).stem
    found = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = frames_dir / f"{stem}{ext}"
        if candidate.exists():
            found = str(candidate)
            break
    
    if found:
        data["image"] = found
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ {json_path.name}: {found}")
    else:
        print(f"❌ {json_path.name}: картинка не найдена")

print("\n🎉 Готово!")
