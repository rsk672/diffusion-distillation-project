# diffusion-distillation

## Методы

В `methods/consistency_models` расположен информация код для обучения EDM и её дистилляции в Consistency Model.

В `methods/dmd2` находится код и подробная информация о запусках метода DMD2.

В `examples/` расположены примении генерации Consistency Model до и после интеграции с методом DMD2.

## Данные

Предусмотрен расчёт метрик для датасетов MS-COCO-2014 и Laion Aesthethics.

Скачать используемый срез датасета MS-COCO-2014 можно по ссылке: https://disk.yandex.ru/d/mYQYI6rJAgvyJA.

Используемый срез датасета Laion Aesthethics можно загрузить с помощью скрипта `data/laion/script.py`.

## Инференс

### DMD2

В папке inference/dmd2 находится скрипт для инференса модели, полученной в ходе дистилляции модели StableDiffusion v1.5 (основано на репозитории https://github.com/tianweiy/DMD2/tree/main). Скачать используемый чекпоинт можно с помощью команды:

`wget https://huggingface.co/tianweiy/DMD2/blob/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000/pytorch_model.bin?download=true -o $OUTPUT_PATH`

### BK-SDM

В папке inference/bk-sdm находится скрипт для трёх конфигураций BK-SDM: bk-sdm-base, bk-sdm-small и bk-sdm-tiny. Все они являются дистилляцией StableDiffusion v1.4. Для выбора конкретной конфигурации нужно при запуске скрипта задать соответствующее значение в аргументе `model_id`, например для bk-sdm-base:

`python inference.py --model_id nota-ai/bk-sdm-base --save_dir $SAVE_DIR_PATH --captions_file $CAPTIONS_PATH`

## Метрики

В директории metrics лежат скрипты для подсчёта метрик FID, CLIP, ImageReward и ImageDiversity. Скрипты принимают на вход директории с изображениями.

Пример подсчёта FID:

`python metrics/calculate_fid.py --fake-dir $MODEL_GENERATED_IMAGES_DIR --gt-dir $GROUND_TRUTH_IMAGES_DIR`

Пример подсчёта CLIP:

`python metrics/calculate_clip.py --image-dir $IMAGES_PATH --captions-file $CAPTIONS_PATH  --how-many 5000`