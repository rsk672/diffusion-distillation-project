# diffusion-distillation

## Данные

На данный момент метрики подсчитаны для датасетов MS-COCO-2014 и Laion Aesthethics.

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

Пример подсчёта ImageReward:

`python metrics/calculate_image_reward.py --image-dir ~/BK-SDM/results/bk-sdm-tiny/im512 --captions-file ~/BK-SDM/captions.txt `

Пример подсчёта ImageDiversity:

`python metrics/calculate_image_diversity.py --image-dir $IMAGE_DIR `

## Методы

### Consistency Models (+EDM)

Для обучения и дистилляции нужен датасет CIFAR-10 в специальном формате, скачать его можно по ссылке: https://drive.google.com/file/d/1ISyZoS5Q0_wY2j_gX0as_AKrMJzbHPs2/view?usp=sharing.

Пример запуска обучения EDM (из директории methods/consistency_models/scripts):

`python edm_train.py --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 32 --lr 0.0001 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --data_dir $DATA_PATH`

Пример дистилляции в Consistency Model (из той же директории):

`python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm l2 --lr_anneal_steps 0 --teacher_model_path $EDM_MODEL_PATH --attention_resolutions 16 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 32 --lr 0.000008 --num_channels 128 --num_head_channels 32 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir $DATA_PATH`

### DMD2 

В начале нужно сгенерировать датасет с изображениями из CIFAR-10 с помощью скрипта `methods/dmd2/main/data/create_ciar10_lmdb.py`. Далее сгенерированный датасет нужно передавать в запускаемые скрипты.

Можно запускать DMD2 на CIFAR-10 для дистилляции уже обученных EDM, скачать их можно отсюда: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/.

Пример запуска (из директории methods/dmd2):

`python main/edm/train_edm.py --model_id $CHECKPOINT_PATH  --output_path $OUTPUT_PATH --batch_size 128 --wandb_name run-edm-cifar10-vp-default --wandb_entity $WANDB_ENTITY --wandb_project dmd2-consistency --dataset_name cifar10-vp --real_image_path $DATASET_PATH --generator_lr 2e-6 --guidance_lr 2e-6 --dfake_gen_update_ratio 5 --max_checkpoint 200 --initialie_generator --delete_ckpts`

Пример запуска с Consistency Model в качестве генератора:

`python main/consistency/train_consistency.py --consistency_model_path $CONSISTENCY_MODEL_PATH --teacher_model_path $EDM_MODEL_PATH --output_path $OUTPUT_PATH --batch_size 128 --wandb_name dmd2-consistency-integrated --wandb_entity $WANDB_ENTITY --wandb_project dmd2-consistency --dataset_name cifar10 --real_image_path $DATASET_PATH --generator_lr 2e-6 --guidance_lr 2e-6 --dfake_gen_update_ratio 5 --max_checkpoint 200 --initialie_generator --delete_ckpts `
