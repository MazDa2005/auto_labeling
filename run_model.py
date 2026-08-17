from merge_model import load_merged_model, run_on_image

# Загружаем модель
device = "cuda:0"
model, names, class_colors = load_merged_model("my_merged_model.pt", device=device)

# Гоняем инференс
result = run_on_image(
    model=model,
    image_path="another_test.jpg",
    out_path="another_out.jpg",
    class_colors=class_colors,
    names=names,
    conf=0.25,
    iou=0.5,
    imgsz=640
)
print(result)
