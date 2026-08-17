"""
sanity_check.py — проверяет главное обещание варианта A: после мёржа confidence
и bbox на СТАРЫХ классах не меняются (совпадают с отдельной старой моделью).

Аналог compare_dicts()/сравнения предсказаний из статьи y-t-g, только для сегментации
и на реальном инференсе, а не на сравнении весов (хотя веса backbone/neck тоже
можно свериться — см. MergedSegModel._sanity_check_shared_backbone, она уже
встроена в merge_model.py и предупреждает автоматически).

Пример:
    python sanity_check.py \
        --old-weights projects/my_project/runs/run_1/weights/best.pt \
        --new-weights runs_new_head/welding_v1/weights/best.pt \
        --test-image test.jpg \
        --conf 0.25
"""
import argparse

import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

from concat_segment_head import ConcatSegmentHead, run_shared_backbone
from merge_model import MergedSegModel, preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-weights", required=True)
    p.add_argument("--new-weights", required=True)
    p.add_argument("--test-image", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--tolerance", type=float, default=1e-4,
                    help="Допустимая макс. разница в bbox/conf (float32 округления)")
    args = p.parse_args()

    import cv2
    img = cv2.imread(args.test_image)
    if img is None:
        raise ValueError(f"Не удалось прочитать {args.test_image}")

    merged = MergedSegModel(args.old_weights, args.new_weights, device=args.device)
    tensor, ratio, pad = preprocess(img, args.imgsz, merged.stride)
    tensor = tensor.to(merged.device)

    # --- старая модель отдельно ---
    with torch.no_grad():
        y_cache = run_shared_backbone(merged.seq_backbone, merged.save_indices, tensor)
        head_input = [y_cache[j] for j in merged.head_old.f]
        (y_old, _proto_old), _ = merged.head_old(head_input)
        (y_merged, _proto_merged), _ = merged.forward(tensor)

    det_old_alone = non_max_suppression(y_old, conf_thres=args.conf, iou_thres=args.iou, nc=merged.nc1)[0]
    det_merged = non_max_suppression(y_merged, conf_thres=args.conf, iou_thres=args.iou,
                                      nc=merged.nc1 + merged.nc2)[0]
    det_merged_old_subset = det_merged[det_merged[:, 5] < merged.nc1]

    print(f"Старая модель отдельно: {len(det_old_alone)} детекций")
    print(f"Merged (только старые классы): {len(det_merged_old_subset)} детекций")

    if len(det_old_alone) != len(det_merged_old_subset):
        print("[WARN] Количество детекций отличается. Это ожидаемо, если новая голова "
              "необучена/зашумлена и её детекции \"вытесняют\" старые из top max_det=300 "
              "по общему списку — увеличьте --conf или max_det в non_max_suppression, "
              "либо (в норме, с обученной новой головой) такого быть не должно.")

    a = det_old_alone[:, :6].cpu().numpy()
    b = det_merged_old_subset[:, :6].cpu().numpy()
    n = min(len(a), len(b))
    if n == 0:
        print("Нет детекций для сравнения — проверьте --conf/--test-image.")
        return

    a_sorted = a[a[:, 4].argsort()][-n:]
    b_sorted = b[b[:, 4].argsort()][-n:]
    max_diff = abs(a_sorted - b_sorted).max()

    print(f"\nМакс. разница по [x1,y1,x2,y2,conf,cls] среди {n} сопоставленных детекций: {max_diff:.6f}")
    if max_diff <= args.tolerance:
        print("✅ Совпадает в пределах допуска — мёрдж не влияет на старые классы.")
    else:
        print("❌ Разница выше допуска — проверьте, что --new-weights обучались через "
              "train_new_head.py с заморозкой поверх ИМЕННО --old-weights.")


if __name__ == "__main__":
    main()

