from ultralytics import YOLO

m = YOLO("/home/iet/iet-share/auto_labeling/projects/video_s_krana/runs/run_1/weights/best.pt")
print("task:", m.task)                  # должно быть 'segment', а не 'detect'
print("names:", m.names)
print("has segment head:", hasattr(m.model, 'model') and any('Segment' in str(type(x)) for x in m.model.model))

r = m("testing.jpg")[0]
md = r.masks.data.cpu().numpy()
print(md.shape, md.min(), md.max(), int((md > 0.5).sum()))