# DMD2

Здесь содержится код для интеграции Consistency Models в метод DMD2 в роли генератора, а также для прямой дистилляции EDM, на основании которой была получена Consistency Model. Код расположен в `main/consistency`. Для обучения соответствующих EDM и Consistency Models следует использовать [данный код](https://github.com/rsk672/diffusion-distillation-project/tree/main/methods/consistency_models).

Также предусмотрена возможность работать с официальной реализацией EDM (чекпоинты этих моделей можно скачать [здесь](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/)). Код расположен в `main/edm`.

Ниже представлены примеры запуска дистилляции в разных конфигурациях. Подразумевается, что они запускаются из текущей директории.

## Окружение

Для запуска кода следует создать conda-окружение на основании файла `dmd2.yml`:

`conda env create -f dmd2.yml -n dmd2`

Далее его необходимо активировать:

`conda activate dmd2`

И в активированном окружении выполнить команду:

`python setup.py develop`

## Входные данные

В рамках экспериментов работа велась с датасетом CIFAR-10. Для обучения используется версия датасета в формате LMDB, скачать её можно [по ссылке](https://disk.yandex.ru/d/GvBAL2uzIpiX-Q). Можно и получить датасет в нужном формате вручную - для этого можно воспользоваться [скриптом](https://github.com/rsk672/diffusion-distillation-project/blob/main/methods/dmd2/main/data/create_cifar10_lmdb.py), который принимает на вход директорию с изображениями и .csv файл, их описывающий (стандартный формат датасета CIFAR-10).

Для подсчета метрики FID в процессе обучения необходим файл с заранее предпосчитанными статистиками на тестовой части CIFAR-10, скачать его можно [здесь](https://disk.yandex.ru/d/Z4jiAlyhZTv_oA). Также необходимо скачать чекпоинт модели InceptionV3, на основании выходов которой считаются необходимые признаки. Сделать это можно с помощью команды `wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl -O $PATH/inception-2015-12-05.pkl`. 

## Запуск DMD2 с EDM

Дистиллировать EDM в одношаговый генератор с помощью метода DMD2 можно с помощью следующего запуска:

`python main/consistency/train_consistency.py --teacher_model_path $EDM_MODEL_PATH --output_path $OUTPUT_PATH --batch_size 128 --wandb_name $WANDB_RUN_NAME --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT --dataset_name cifar10 --real_image_path $LMDB_DATASET_PATH --generator_lr 2e-6 --guidance_lr 2e-6 --dfake_gen_update_ratio 5 --max_checkpoint 200 --initialie_generator --delete_ckpts --num_train_timesteps 40 --ref_path $REF_STATS_PATH --detector_url $INCEPTION_MODEL_PATH`

`$EDM_MODEL_PATH` - путь до обученной EDM

`$OUTPUT_PATH` - путь до директории, куда будут сохранены чекпоинты

`$WANDB_RUN_NAME` - название запуска в wandb

`$WANDB_PROJECT` - название проекта в wandb

`$LMDB_DATASET_PATH` - путь до скачанного датасета в формате LMDB

`$REF_STATS_PATH` - путь до файла со статистиками для FID, посчитанными на тестовой части датасета

`$INCEPTION_MODEL_PATH` - путь до чекпоинта модели InceptionV3, используемой для подсчета FID

## Запуск DMD2 с Consistency Model

Запууск метода DMD2 с обученной Consistency Model в качестве генератора осуществляется следующим образом:

`python main/consistency/train_consistency.py --consistency_model_path $CONSISTENCY_MODEL_PATH --teacher_model_path $EDM_MODEL_PATH --output_path $OUTPUT_PATH --batch_size 128 --wandb_name $WANDB_RUN_NAME --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT --dataset_name cifar10 --real_image_path $LMDB_DATASET_PATH --generator_lr 2e-6 --guidance_lr 2e-6 --dfake_gen_update_ratio 5 --max_checkpoint 200 --initialie_generator --delete_ckpts --num_train_timesteps 40 --ref_path $REF_STATS_PATH --detector_url $INCEPTION_MODEL_PATH`

`$CONSISTENCY_MODEL_PATH` - путь до обученной Consistency Model

`$EDM_MODEL_PATH` - путь до обученной EDM

`$OUTPUT_PATH` - путь до директории, куда будут сохранены чекпоинты

`$WANDB_RUN_NAME` - название запуска в wandb

`$WANDB_PROJECT` - название проекта в wandb

`$LMDB_DATASET_PATH` - путь до скачанного датасета в формате LMDB

`$REF_STATS_PATH` - путь до файла со статистиками для FID, посчитанными на тестовой части датасета

`$INCEPTION_MODEL_PATH` - путь до чекпоинта модели InceptionV3, используемой для подсчета FID

## Запуск DMD2 с GAN-лоссом

При необходимости использовать конфигурацию с GAN-лоссом, к команде запуска обучения следует добавить следующие аргументы:

`$--gan_classifier` - флаг, указывающий на использование GAN-лосса

`$--gen_cls_loss_weight VALUE` - значение коэффициента перед GAN-лоссом

`$--diffusion_gan` - флаг, указывающий на использование диффузионной модели в качестве дискриминатора

`$--diffusion_gan_max_timestep VALUE` - максимальное значение временного шага, используемое при случайном семплинге в процессе обучения (здесь стоит ориентироваться на значение параметра `--num_train_timesteps`)

## Запуск DMD2 с предобученной EDM из официального репозитория

[По ссылке](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/) находятся чекпоинты предобученных моделей от авторов [оригинальной статьи про EDM](https://arxiv.org/pdf/2206.00364). 

Пример запуска с чекпоинтом `edm-cifar10-32x32-uncond-vp.pkl`:

`python main/edm/train_edm.py --model_id $CHECKPOINT_PATH/edm-cifar10-32x32-cond-vp.pkl --output_path $OUTPUT_PATH --batch_size 128 --wandb_name $WANDB_RUN_NAME --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT --dataset_name cifar10-vp --real_image_path $LMDB_DATASET_PATH --generator_lr 2e-6 --guidance_lr 2e-6 --dfake_gen_update_ratio 5 --max_checkpoint 200 --initialie_generator --ref_path $REF_STATS_PATH --detector_url $INCEPTION_MODEL_PATH`

`$OUTPUT_PATH` - путь до директории, куда будут сохранены чекпоинты

`$WANDB_RUN_NAME` - название запуска в wandb

`$WANDB_PROJECT` - название проекта в wandb

`$LMDB_DATASET_PATH` - путь до скачанного датасета в формате LMDB

`$REF_STATS_PATH` - путь до файла со статистиками для FID, посчитанными на тестовой части датасета

`$INCEPTION_MODEL_PATH` - путь до чекпоинта модели InceptionV3, используемой для подсчета FID

## Чекпоинты

Чекпоинт EDM, на основании которой производилась дистилляция в Consistency Model, можно скачать [здесь](https://disk.yandex.ru/d/NlhpRu8i79___w).

Чекпоинт Consistency Model, полученный в ходе дистилляции EDM, можно скачать [здесь](https://disk.yandex.ru/d/PrKIM2w9nEgy1g).

Чекпоинт Consistency Model, полученный после интеграции в метод DMD2 в роли генератора, можно скачать [по ссылке](https://disk.yandex.ru/d/BYQJQPr44GhtJA). 