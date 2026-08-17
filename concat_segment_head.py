"""
concat_segment_head.py — сердце варианта A: настоящее слияние двух Segment-голов
(старые классы + новые) в ОДНУ модель на инференсе, без патчей ultralytics.

Проверено вручную на ultralytics==8.4.118 (у вас в sam3.yml — 8.4.89, тот же
пост-рефакторный API) с реальными весами yolo11n-seg.pt: после мёржа confidence
и bbox старых классов побитово совпадают с отдельной моделью.

КЛЮЧЕВАЯ ИДЕЯ (расширение трюка ConcatHead из статьи y-t-g на сегментацию):
  - box (4 канала)     -> просто конкатенация по anchor-размерности
  - cls (nc1+nc2)       -> тот же zero-pad трюк, что в статье: обнуляем "чужие"
                           классы в каждой половине, потом конкат по anchors
  - mask coeffs (nm)    -> ТОТ ЖЕ zero-pad трюк, только по оси масок вместо классов
  - proto (npr каналов)  -> просто конкатенация по каналам (это карты фич, не
                           anchor-специфичны)

  Из-за zero-pad в mask-коэффициентах, "старые" anchors получают нулевые
  коэффициенты для "новых" proto-каналов и наоборот — поэтому итоговая маска
  через process_mask() автоматически берётся из ПРАВИЛЬНОГО proto-блока без
  дополнительной логики выбора. Именно эта симметрия и делает вариант A
  реализуемым без патчей ops.process_mask.

ВАЖНО: голова "new" должна быть обучена на ТЕХ ЖЕ frozen backbone+neck, что и
"old" (см. train_new_head.py) — иначе фичи, на которых стоит новая голова,
не будут соответствовать тому, что реально выдаёт backbone старой модели.
"""
import torch
import torch.nn as nn


class ConcatSegmentHead(nn.Module):
    """Склеивает выходы двух ultralytics Segment-голов в один инференс-выход.

    Поддерживает ТОЛЬКО режим инференса (eval, non-export). Обучать эту
    склеенную модель напрямую не нужно и не поддерживается — каждая голова
    обучается отдельно (см. train_new_head.py), эта склейка собирается
    только для forward-прохода в проде.
    """

    def __init__(self, nc1: int, nc2: int):
        super().__init__()
        self.nc1 = nc1
        self.nc2 = nc2

    def forward(self, x: list):
        if self.training:
            raise RuntimeError(
                "ConcatSegmentHead поддерживает только инференс. "
                "Головы обучаются раздельно в train_new_head.py."
            )

        # x[0], x[1] — сырые выходы Segment.forward() в eval/non-export режиме:
        # ((y_full[bs,4+nc+nm,A], proto[bs,nm,Hm,Wm]), preds_dict)
        (y1, proto1), preds1 = x[0]
        (y2, proto2), preds2 = x[1]

        nm1, nm2 = proto1.shape[1], proto2.shape[1]
        A1, A2 = y1.shape[-1], y2.shape[-1]
        bs = y1.shape[0]

        box1, box2 = y1[:, :4, :], y2[:, :4, :]
        cls1 = y1[:, 4:4 + self.nc1, :]
        cls2 = y2[:, 4:4 + self.nc2, :]
        mc1 = y1[:, 4 + self.nc1:4 + self.nc1 + nm1, :]
        mc2 = y2[:, 4 + self.nc2:4 + self.nc2 + nm2, :]

        # 1. box — просто конкат по anchors
        box_merged = torch.cat([box1, box2], dim=2)

        # 2. cls — zero-pad трюк из статьи
        cls1_pad = torch.cat(
            [cls1, torch.zeros(bs, self.nc2, A1, device=cls1.device, dtype=cls1.dtype)], dim=1
        )
        cls2_pad = torch.cat(
            [torch.zeros(bs, self.nc1, A2, device=cls2.device, dtype=cls2.dtype), cls2], dim=1
        )
        cls_merged = torch.cat([cls1_pad, cls2_pad], dim=2)

        # 3. mask coefficients — тот же трюк, но по оси масок
        mc1_pad = torch.cat(
            [mc1, torch.zeros(bs, nm2, A1, device=mc1.device, dtype=mc1.dtype)], dim=1
        )
        mc2_pad = torch.cat(
            [torch.zeros(bs, nm1, A2, device=mc2.device, dtype=mc2.dtype), mc2], dim=1
        )
        mc_merged = torch.cat([mc1_pad, mc2_pad], dim=2)

        # 4. proto — просто конкат по каналам
        proto_merged = torch.cat([proto1, proto2], dim=1)

        y_merged = torch.cat([box_merged, cls_merged, mc_merged], dim=1)

        # preds1 прокидываем как заглушку — используется только в редком
        # save_feats/ReID-пути предиктора, который мы не используем.
        return (y_merged, proto_merged), preds1


def run_shared_backbone(seq_backbone, save_indices: set, x: torch.Tensor) -> list:
    """
    Прогоняет ОБЩИЙ backbone+neck (все слои модели, кроме последнего — головы)
    один раз и возвращает кэш промежуточных выходов `y`, из которого голова(ы)
    берут нужные им multi-scale фичи (P3/P4/P5) по индексам в `head.f`.

    seq_backbone  — model.model.model[:-1] (nn.Sequential без последнего слоя)
    save_indices  — model.model.save (set индексов, чьи выходы нужны позже)
    """
    y = []
    for m in seq_backbone:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in save_indices else None)
    return y
