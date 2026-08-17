# dual_head_seg — инкрементальное добавление классов к student YOLO-seg

**Основной путь (вариант A) — настоящее слияние в ОДНУ модель**, реализация трюка
[y-t-g: Extending YOLOv8 With New Classes](https://y-t-g.github.io/tutorials/yolov8n-add-classes/)
для **instance segmentation**, проверенная вручную на реальном ultralytics
(8.4.118, у вас в `sam3.yml` — 8.4.89, тот же пост-рефакторный API) с реальными
весами и реальной картинкой: **confidence и bbox старых классов после мёржа
побитово совпадают** (`sanity_check.py` печатает `max_diff=0.000000`).

## Идея

1. `train_new_head.py` — учим новую `Segment`-голову **поверх замороженного
   backbone/neck старой модели** (не generic pretrained checkpoint!). Как в
   статье: `freeze=N`, BN-слои принудительно в eval() каждую эпоху (иначе BN
   running stats всё равно "уедут", даже с requires_grad=False), ultralytics
   сам переинициализирует только последний слой под новое число классов.
2. `concat_segment_head.py` — `ConcatSegmentHead`: расширение ConcatHead-трюка
   из статьи на маски. Ключевое наблюдение (в статье этого нет, она только
   про detection): тот же zero-pad трюк, что применяется к class-логитам,
   применяется и к mask-коэффициентам, а proto-тензоры просто конкатенируются
   по каналам. Из-за zero-pad "старые" anchors получают нулевые коэффициенты
   для "новых" proto-каналов и наоборот — поэтому `ops.process_mask()`
   автоматически берёт маску из правильного proto-блока **без каких-либо
   патчей** в ultralytics. Патчить `tasks.py`/`ops.py` не нужно вообще.
3. `merge_model.py` — собирает `MergedSegModel` (общий backbone/neck + два
   head'а + `ConcatSegmentHead`) программно, в питоне, БЕЗ yaml-конфигурации
   и БЕЗ патча `parse_model` (в отличие от оригинальной статьи — там патч
   `tasks.py` был нужен из-за yaml-driven архитектуры; текущий ultralytics
   сильно отличается от версии в статье, поэтому мы просто напрямую берём
   готовые `Segment`-модули из двух чекпоинтов и гоняем forward руками).
   Делает полный инференс: letterbox → shared backbone → 2 головы →
   ConcatSegmentHead → NMS → process_mask → отрисовка.
4. `sanity_check.py` — автоматическая проверка "старые классы не изменились".

## Как использовать

```bash
# 1. Обучить новую голову поверх текущего student (frozen backbone)
python train_new_head.py \
    --old-weights projects/my_project/runs/run_1/weights/best.pt \
    --dataset-dir dataset_new_classes/ \
    --classes-file new_classes.json \
    --runs-dir runs_new_head --run-name welding_v1 --epochs 100

# 2. Проверить, что мёрдж не сломал старые классы
python sanity_check.py \
    --old-weights projects/my_project/runs/run_1/weights/best.pt \
    --new-weights runs_new_head/welding_v1/weights/best.pt \
    --test-image any_frame.jpg

# 3. Инференс объединённой моделью
python merge_model.py \
    --old-weights projects/my_project/runs/run_1/weights/best.pt \
    --new-weights runs_new_head/welding_v1/weights/best.pt \
    --classes-file merged_classes.json \
    --input test.jpg --output out.jpg
```

`merged_classes.json` — объединённый classes.json (старые классы как есть +
новые с индексами, продолженными после старых). Соберите руками — просто
сконкатенируйте два списка классов с пересчётом `index` (старые с 0, новые
начиная с `len(старые)`).

## Важные допущения и границы применимости

- **Backbone новой головы должен быть буквально тем же (числа побитово
  совпадают)**, что backbone старой модели — это гарантируется только если
  `train_new_head.py` обучал голову с `freeze` поверх ИМЕННО `--old-weights`.
  `MergedSegModel` при инициализации сверяет это автоматически и печатает
  `[WARN]`, если веса разошлись.
- Один "раунд" мёржа = одна новая голова. Если понадобится добавить ещё
  классов позже — можно либо обучить третью голову (расширить
  `ConcatSegmentHead` до N голов — механика та же, просто больше слагаемых
  в zero-pad конкатенации), либо смёржить старую объединённую модель как
  новый "old" и обучить очередную голову поверх неё.
- Экспорт в ONNX/TensorRT **не поддержан** — вся механика завязана на
  `Segment.forward()` в eval/non-export режиме (сырые Python tuple/dict).
  Если понадобится экспорт для деплоя — это отдельная, более объёмная задача
  (нужно писать собственный экспортируемый forward без internal dict-путей).
  Для вашего текущего деплоя (`.pt` через `ultralytics.YOLO()` в Docker) это
  не проблема.
- Проверено на ultralytics 8.4.118. Внутренняя структура `Segment.forward()`
  (`dict`-based `forward_head`, `((y, proto), preds)` tuple) — деталь
  реализации конкретной версии; если апгрейднете ultralytics и `sanity_check.py`
  начнёт показывать расхождение — вероятно, поменялась внутренняя структура,
  нужно свериться заново (`inspect.getsource(Segment.forward)`).

## Альтернатива (вариант B, оставлена как fallback)

`dual_model_infer.py` + `benchmark_dual.py` — две ПОЛНОСТЬЮ отдельные модели,
гоняются независимо, результаты просто конкатенируются (без общего backbone).
Проще, но 2x forward pass вместо одного. Пригодится, если warning из
`MergedSegModel` покажет, что backbone у вас всё-таки разошёлся (например,
случайно забыли `freeze`), либо для диагностики "вариант A ведёт себя не так,
как ожидалось — посмотрим, что скажут раздельные модели".

