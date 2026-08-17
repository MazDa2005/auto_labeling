"""
merge_model.py — ядро merged YOLO-seg модели.
Используется build_merged_model.py и run_merged_model.py.
"""
import colorsys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from concat_segment_head import ConcatSegmentHead, run_shared_backbone


# ── Сборка модели ─────────────────────────────────────────────────────────────

class MergedSegModel(nn.Module):
    """Общий backbone/neck от старой модели + две головы + ConcatSegmentHead."""

    def __init__(self, old_weights: str, new_weights: str, verbose: bool = True):
        super().__init__()
        yolo_old = YOLO(old_weights)
        yolo_new = YOLO(new_weights)
        model_old = yolo_old.model.eval()
        model_new = yolo_new.model.eval()

        # Проверка, что обе модели — segment
        self._check_segment_models(yolo_old, yolo_new)

        # Предупреждение о разном масштабе
        self._check_architecture_compatibility(yolo_old, yolo_new, verbose)

        # Имена классов — прямо из весов
        self.names_old: list[str] = [yolo_old.names[i] for i in range(len(yolo_old.names))]
        self.names_new: list[str] = [yolo_new.names[i] for i in range(len(yolo_new.names))]

        self.seq_backbone = model_old.model[:-1]
        self.head_old = model_old.model[-1]
        self.head_new = model_new.model[-1]

        self.save_indices = set(model_old.save)
        self.stride = int(model_old.stride.max())

        self.nc1 = self.head_old.nc
        self.nc2 = self.head_new.nc
        self.concat_head = ConcatSegmentHead(nc1=self.nc1, nc2=self.nc2).eval()

        self._sanity_check_shared_backbone(model_old, model_new, verbose)

    @property
    def names(self) -> list[str]:
        """Полный список имён: сначала старые классы, потом новые."""
        return self.names_old + self.names_new

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _check_segment_models(yolo_old: YOLO, yolo_new: YOLO):
        # В YOLOv11 сегментационная голова называется Segment.
        # Проверяем по официальному атрибуту task и по имени последнего слоя.
        old_is_seg = (yolo_old.task == 'segment' or 
                      type(yolo_old.model.model[-1]).__name__ == 'Segment')
        new_is_seg = (yolo_new.task == 'segment' or 
                      type(yolo_new.model.model[-1]).__name__ == 'Segment')

        if not (old_is_seg and new_is_seg):
            raise ValueError(
                "Обе модели должны быть segment. "
                f"old: task={yolo_old.task}, последний слой={type(yolo_old.model.model[-1]).__name__}\n"
                f"new: task={yolo_new.task}, последний слой={type(yolo_new.model.model[-1]).__name__}"
            )

    @staticmethod
    def _check_architecture_compatibility(yolo_old: YOLO, yolo_new: YOLO, verbose: bool):
        old_params = sum(p.numel() for p in yolo_old.model.parameters())
        new_params = sum(p.numel() for p in yolo_new.model.parameters())
        ratio = max(old_params, new_params) / min(old_params, new_params)
        if ratio > 1.5 and verbose:
            print(
                f"[WARN] Модели разного масштаба: "
                f"old={old_params / 1e6:.1f}M, new={new_params / 1e6:.1f}M. "
                f"Backbone может не совпасть."
            )

    def _sanity_check_shared_backbone(self, model_old, model_new, verbose: bool):
        mismatches = []
        old_sd = dict(model_old.state_dict())
        new_sd = dict(model_new.state_dict())
        for k in old_sd:
            if k.startswith(f"model.{self.head_old.i}."):
                continue
            if k in new_sd and old_sd[k].shape == new_sd[k].shape:
                if not torch.equal(old_sd[k].cpu(), new_sd[k].cpu()):
                    mismatches.append(k)
        if mismatches:
            msg = (
                f"[WARN] backbone/neck разошлись в {len(mismatches)} тензорах "
                f"(например: {mismatches[:3]}). Убедитесь, что --new-weights обучались "
                f"через train_new_head.py поверх --old-weights."
            )
            if verbose:
                print(msg)
                for k in mismatches[:10]:
                    print(f"  - {k}")
            else:
                print(msg)

    @torch.no_grad()
    def forward(self, img_tensor: torch.Tensor):
        x = img_tensor.to(self.device)
        y_cache = run_shared_backbone(self.seq_backbone, self.save_indices, x)
        head_input = [y_cache[j] for j in self.head_old.f]
        out_old = self.head_old(head_input)
        out_new = self.head_new(head_input)
        return self.concat_head([out_old, out_new])

    def save_merged(self, path: str, metadata: dict | None = None):
        """Сохраняет модель в .pt — имена классов уже внутри, JSON не нужен."""
        # Создаём папку, если её нет
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.cpu(),
            "names": self.names,
            "names_old": self.names_old,
            "names_new": self.names_new,
            "stride": self.stride,
            "nc1": self.nc1,
            "nc2": self.nc2,
            "save_indices": sorted(list(self.save_indices)),
            "metadata": metadata or {},
        }
        torch.save(state, path)
        print(f"[save] Модель сохранена: {path}  ({self.nc1}+{self.nc2} классов)")


def load_merged_model(ckpt_path: str, device: str = "cpu"):
    """Загружает сохранённую merged-модель. Возвращает (model, names, class_colors)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model: MergedSegModel = ckpt["model"].to(device).eval()
    names: list[str] = ckpt["names"]
    class_colors = build_class_colors(names)
    print(
        f"[load] {ckpt_path}: {ckpt['nc1']}+{ckpt['nc2']} классов → "
        f"{len(names)} имён"
    )
    return model, names, class_colors


class MergedExportWrapper(nn.Module):
    """Обёртка для экспорта в ONNX (только тензоры на выходе)."""

    def __init__(self, model: MergedSegModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        (y_merged, proto_merged), _ = self.model(x)
        return y_merged, proto_merged


def export_to_onnx(model: MergedSegModel, path: str, imgsz: int = 640):
    wrapper = MergedExportWrapper(model).to(model.device).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=model.device)
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=["images"],
        output_names=["y_merged", "proto"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "y_merged": {0: "batch", 1: "anchors"},
            "proto": {0: "batch", 2: "mask_h", 3: "mask_w"},
        },
        opset_version=17,
    )
    print(f"[onnx] Экспортировано: {path}")


# ── Цвета ─────────────────────────────────────────────────────────────────────

def generate_colors(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def _parse_color(value) -> tuple[int, int, int] | None:
    """Парсит hex/RGB в BGR."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) == 6:
            try:
                r = int(s[0:2], 16)
                g = int(s[2:4], 16)
                b = int(s[4:6], 16)
                return (b, g, r)
            except ValueError:
                pass
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            r = int(value[0])
            g = int(value[1])
            b = int(value[2])
            return (b, g, r)
        except (TypeError, ValueError):
            pass
    return None


def build_class_colors(
    names: list[str],
    data_yaml_path: str | None = None,
    classes_file: str | None = None,
) -> dict[str, tuple[int, int, int]]:
    """
    Берёт цвета из data.yaml (поле colors), потом из classes.json (fallback),
    потом генерирует автоматически.
    """
    colors_from_yaml: dict[str, tuple[int, int, int]] = {}

    if data_yaml_path and Path(data_yaml_path).exists():
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("colors", {})
        if isinstance(raw, dict):
            for k, v in raw.items():
                parsed = _parse_color(v)
                if parsed is not None:
                    colors_from_yaml[str(k).strip()] = parsed

    # Fallback на classes.json (с поддержкой пробелов в ключах)
    if not colors_from_yaml and classes_file and Path(classes_file).exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes_list = data.get("classes", data.get("classes ", []))
        for c in classes_list:
            name = str(c.get("name", c.get("name ", ""))).strip()
            color = str(c.get("color", c.get("color ", ""))).strip()
            parsed = _parse_color(color)
            if name and parsed is not None:
                colors_from_yaml[name] = parsed

    palette = generate_colors(len(names))
    return {
        name: colors_from_yaml.get(name, palette[i])
        for i, name in enumerate(names)
    }


# ── Препроцессинг ─────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, new_shape: int = 640, stride: int = 32):
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape - new_unpad[0]) % stride
    dh = (new_shape - new_unpad[1]) % stride
    dw /= 2
    dh /= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, r, (left, top)


def preprocess(img_bgr: np.ndarray, imgsz: int, stride: int):
    padded, ratio, pad = letterbox(img_bgr, imgsz, stride)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor, ratio, pad


def scale_boxes_back(boxes: torch.Tensor, ratio: float, pad: tuple, orig_shape: tuple):
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_shape[0])
    return boxes


# ── Отрисовка ─────────────────────────────────────────────────────────────────

def _draw_mask(img: np.ndarray, mask_array: np.ndarray, color, alpha: int = 110):
    """Полупрозрачная маска + контур."""
    colored = np.zeros_like(img)
    colored[mask_array] = color
    blended = cv2.addWeighted(colored, alpha / 255, img, 1 - alpha / 255, 0)
    img[mask_array] = blended[mask_array]

    mask_u8 = mask_array.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 2)


def _draw_detection(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + 4), color, -1)
    cv2.putText(
        img, label, (x1 + 3, ty + th),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )


# ── Инференс ──────────────────────────────────────────────────────────────────

def run_inference(
    model: MergedSegModel,
    img: np.ndarray,
    names: list[str],
    class_colors: dict,
    conf: float = 0.25,
    iou_thres: float = 0.5,
    imgsz: int = 640,
) -> tuple[list[dict], float]:
    """
    Инференс на одном кадре (np.ndarray, BGR).
    Возвращает (detections, inference_ms). Маски рисуются прямо на img.
    """
    if img is None:
        raise ValueError("Пустой кадр")

    orig_h, orig_w = img.shape[:2]
    tensor, ratio, pad = preprocess(img, imgsz, model.stride)
    tensor = tensor.to(model.device)

    t0 = time.time()
    (y_merged, proto_merged), _ = model(tensor)
    det = non_max_suppression(
        y_merged,
        conf_thres=conf,
        iou_thres=iou_thres,
        nc=model.nc1 + model.nc2,
    )[0]
    inference_ms = (time.time() - t0) * 1000

    detections = []
    if det.shape[0] > 0:
        masks = ops.process_mask(
            proto_merged[0],
            det[:, 6:],
            det[:, :4],
            tensor.shape[2:],
            upsample=True,
        )
        boxes = scale_boxes_back(det[:, :4], ratio, pad, (orig_h, orig_w))

        for i in range(det.shape[0]):
            cls_id = int(det[i, 5])
            cls_name = names[cls_id] if cls_id < len(names) else str(cls_id)
            conf_score = float(det[i, 4])
            x1, y1, x2, y2 = boxes[i].int().tolist()
            color = class_colors.get(cls_name, (150, 150, 150))

            # Правильная обрезка маски (padding несимметричный)
            mask_np = masks[i].cpu().numpy()
            mh, mw = mask_np.shape
            crop_top, crop_left = int(pad[1]), int(pad[0])
            orig_h_scaled = int(orig_h * ratio)
            orig_w_scaled = int(orig_w * ratio)
            mask_crop = mask_np[
                crop_top:crop_top + orig_h_scaled,
                crop_left:crop_left + orig_w_scaled,
            ]
            mask_full = cv2.resize(mask_crop, (orig_w, orig_h)) > 0.5

            _draw_mask(img, mask_full, color)
            _draw_detection(img, x1, y1, x2, y2, f"{cls_name} {conf_score:.2f}", color)

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf_score, 4),
                "bbox": [x1, y1, x2, y2],
            })

    return detections, inference_ms


def run_on_image(
    model: MergedSegModel,
    image_path: str,
    out_path: str,
    names: list[str],
    class_colors: dict,
    conf: float = 0.25,
    iou_thres: float = 0.5,
    imgsz: int = 640,
) -> dict:
    """Инференс на картинке."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {image_path}")

    detections, inference_ms = run_inference(
        model, img, names, class_colors, conf, iou_thres, imgsz
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(out_path, img):
        raise ValueError(f"Не удалось сохранить результат: {out_path}")

    return {
        "detections": detections,
        "inference_ms": round(inference_ms, 1),
        "n_detections": len(detections),
        "output_path": out_path,
    }


def run_on_video(
    model: MergedSegModel,
    video_path: str,
    out_path: str,
    names: list[str],
    class_colors: dict,
    conf: float = 0.25,
    iou_thres: float = 0.5,
    imgsz: int = 640,
    max_frames: int | None = None,
) -> dict:
    """Инференс на видео."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        print(f"[WARN] FPS не определён, использую {fps}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1

    print(f"Видео: {w}x{h}, FPS={fps}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print("[WARN] mp4v не работает, пробую XVID...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = str(Path(out_path).with_suffix(".avi"))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось инициализировать VideoWriter для {out_path}")
        print(f"Использую XVID, выход: {out_path}")

    frame_times = []
    frame_idx = 0
    total_detections = 0
    skipped_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            skipped_frames += 1

        t0 = time.time()
        dets, _ = run_inference(model, frame, names, class_colors, conf, iou_thres, imgsz)
        frame_times.append(time.time() - t0)
        total_detections += len(dets)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Обработано кадров: {frame_idx}")

    cap.release()
    writer.release()

    if skipped_frames > 0:
        print(f"[WARN] Пропущено/ресайзнуто кадров: {skipped_frames}")

    avg_ms = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0
    return {
        "frames_processed": frame_idx,
        "total_detections": total_detections,
        "avg_inference_ms": round(avg_ms, 1),
        "avg_fps": round(1000 / avg_ms, 1) if avg_ms else 0,
        "output_path": out_path,
    }