# Consistency Models

Здесь содержится код для обучения EDM и её дальнейшей дистилляции в Consistency Model. 

Все скрипты для обучения и семплинга запускаются из директории `scripts/`.

## Окружение

Для запуска кода следует создать conda-окружение на основании файла `consistency_models.yml`:

`conda env create -f dmd2.yml -n consistency_models`

Далее его необходимо активировать:

`conda activate consistency_models`

И в активированном окружении выполнить команду:

`pip install -e .`

## Датасет

В рамках экспериментов работа велась с датасетом CIFAR-10. В нужном формате датасет для обучения можно скачать по [ссылке](https://disk.yandex.ru/d/NdbRhJuvqZ8w2A). Использовалось оригинальное разбиение датасета на обучающую и тестовую часть размерами 50000 и 10000 изображений соответственно.

## Обучение EDM

Обучить EDM можно следующим образом: 

```python edm_train.py --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 32 --lr 0.0001 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --data_dir $DATASET_PATH --logging_dir $LOGGING_DIR```

`$DATASET_PATH` - путь до скачанного датасета CIFAR-10

`$LOGGING_DIR` - директория, куда будут сохранены логи и чекпоинт модели.

## Дистилляция EDM в Consistency Model

Обученную EDM можно дистиллировать в Consistency Model с помощью следующего запуска:

```python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path $TEACHER_MODEL_PATH --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 32 --lr 0.000008 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir $DATASET_PATH --logging_dir $LOGGING_DIR```

`$DATASET_PATH` - путь до скачанного датасета CIFAR-10

`$LOGGING_DIR` - директория, куда будут сохранены логи и чекпоинт модели

`$TEACHER_MODEL_PATH` - путь до чекпоинта обученной EDM

## Семплирование

Представленные ниже скрипты генерируют .npz файл с изображениями. Этот файл можно передать в скрипты для подсчёта метрик качества, располагающихся [здесь](https://github.com/rsk672/diffusion-distillation-project/tree/main/metrics).

Примеры генерации EDM и Consistency Model до и после метода DMD2 можно увидеть [здесь](https://github.com/rsk672/diffusion-distillation-project/tree/main/examples).

### Семплинг из EDM

Осуществляется следующим образом:

```python image_sample.py --training_mode edm --batch_size 500 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path $EDM_MODEL_PATH --attention_resolutions 16  --class_cond True --dropout 0.1 --image_size 32 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --num_samples $NUM_SAMPLES --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras```

`$EDM_MODEL_PATH` - путь до чекпоинта обученной EDM

`$NUM_SAMPLES` - количество сгенерированных семплов

### Семплинг из Consistency Model (одношаговый)

Осуществляется следующим образом:

```python image_sample.py --batch_size 500 --training_mode consistency_distillation --sampler onestep --model_path $CONSISTENCY_MODEL_PATH --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 32 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --num_samples $NUM_SAMPLES --resblock_updown True --use_fp16 True --weight_schedule uniform```

`$CONSISTENCY_MODEL_PATH` - путь до чекпоинта Consistency Model

`$NUM_SAMPLES` - количество сгенерированных семплов

### Семплинг из Consistency Model (многошаговый)

Осуществляется следующим образом:

```python image_sample.py --batch_size 500 --training_mode consistency_distillation --sampler multistep --ts $TIMESTEPS  --model_path $CHECKPOINT_MODEL_PATH --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 32 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --num_samples $NUM_SAMPLES --resblock_updown True --use_fp16 True --weight_schedule uniform```

`$CONSISTENCY_MODEL_PATH` - путь до чекпоинта Consistency Model

`$NUM_SAMPLES` - количество сгенерированных семплов

Важное отличие от одношагового семплинга заключается в наличии аргумента ts. $TIMESTEPS должен представлять собой строку из временных шагов, по которым осуществляется многошаговый семплинг, например `0,19,39`.

## Чекпоинты

Чекпоинт EDM, на основании которой производилась дистилляция в Consistency Model, можно скачать [здесь](https://disk.yandex.ru/d/NlhpRu8i79___w).

Чекпоинт Consistency Model, полученный в ходе дистилляции EDM, можно скачать [здесь](https://disk.yandex.ru/d/PrKIM2w9nEgy1g).

Чекпоинт Consistency Model, полученный после интеграции в метод DMD2 в роли генератора, можно скачать [по ссылке](https://disk.yandex.ru/d/BYQJQPr44GhtJA). 