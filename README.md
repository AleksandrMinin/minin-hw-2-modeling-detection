# Barcode detection

Решение задачи детекции штрих-кодов на изображениях.


### Датасет

Включает 305 изображений штрих-кодов, которые размечены bbox.
Скачать данные можно [отсюда](https://disk.yandex.ru/d/kUkdcBR78Fzoxg).

### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:

    a. Обновить pip
    ```
    pip install --upgrade pip 
    ```
    b. Выполнить команду
    ```
    pip install -r requirements.txt
    ```

3. Настройка ClearML

    a. [В своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем:
      "Settings" -> "Workspace" -> "Create new credentials"
      
    b. Появится инструкция для "LOCAL PYTHON", копируем её.
    
    с. Пишем в консоли `clearml-init` и вставляем конфиг из инструкции.

### Обучение
Запуск тренировки c `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python train.py > log.out
```

Запуск тренировки без `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python train.py
```

### ClearML
Метрики и конфигурации экспериментов:
1. [experiment_1](https://app.clear.ml/projects/2b135481bfd94a21a4b3197ecaf4e486/experiments/acedfe938a0045a9977f2804efb802fc/output/execution)
2. [experiment_2](https://app.clear.ml/projects/2b135481bfd94a21a4b3197ecaf4e486/experiments/fad68c85ddfe468898628e5ae6803fbd/output/execution)
3. [experiment_3](https://app.clear.ml/projects/2b135481bfd94a21a4b3197ecaf4e486/experiments/d11eb703379c427481aed607dc0cf90a/output/execution)

### Тестирование модели
Результаты сегментации лучшей модели можно посмотреть в  /notebooks/check_pred_mask.ipynb

### DVC
#### Добавление модели в DVC
1. Инициализация DVC

    В директории проекта пишем команды:
    ```
    dvc init
    ```
    ```
    dvc remote add --default myremote ssh://91.206.15.25/home/aleksandrminin/dvc_files
    ```

    ```
    dvc remote modify myremote user aleksandrminin
    dvc config cache.type hardlink,symlink
    
    dvc remote modify myremote keyfile /path/to/your/private_key
    ```

    Про типы линков [здесь](https://dvc.org/doc/user-guide/large-dataset-optimization#file-link-types-for-the-dvc-cache).
    

2. Добавление модели в DVC
    
    Копируем в `weights` обученную модель
    ```
    cd weights
    dvc add model.pt
    dvc push
   ```
   Если появится ошибка с правами, можно дополнительно указать путь до приватного ключа:
   ```
   dvc remote modify myremote keyfile /path/to/your/private_key
   ```
   Про генерацию ssh-ключа [здесь](https://selectel.ru/blog/tutorials/how-to-generate-ssh/).

3. Делаем коммит с новой моделью:
    ```
    git add .
    git commit -m "add new model"
   ```

#### Загрузка лучшей модели из DVC к себе
   ```
    git pull origin main
    dvc pull
   ```

### Запуск литера
Из папки с проектом выполнить:
   ```
   python -m pip install wemake-python-styleguide==0.16.1
   flake8 src/
   ```

### Запуск тестов на pytest
Из папки с проектом выполнить:
   ```
   PYTHONPATH=. pytest tests -p no:warnings
   ```
