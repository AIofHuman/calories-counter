# Calories Counter

Проект для многомодального предсказания энергетической ценности блюда (калорий) на основе изображения и состава ингредиентов.

---

## Описание

Этот проект представляет собой задачу регрессии для оценки калорийности блюда с использованием:
- изображения блюда,
- списка ингредиентов (обрабатываемого как мультитэговую разреженную разметку),
- современных CV & NLP подходов (используется EfficientNet для работы с изображениями).

> Для повышения устойчивости применяются аугментации изображений и текста.

---

## Структура проекта

```
├── config/effinet_multihot.yaml    # Настройки модели и обучения
├── data/                          # Папка с изображениями и csv-файлами (dish.csv, ingredients.csv)
├── src/
│   ├── dataset.py                 # Подготовка датасетов
│   ├── infer.py                   # Инференс модели
│   ├── multimodal.py              # Модель и объединение модальностей
│   ├── seed.py                    # Установка сидов/детерминизм
│   ├── train.py                   # Обучение
│   ├── transform.py               # Аугментации
│   └── __init__.py
├── tests/
│   └── test_transform.py          # Тесты аугментаций
├── requirements.txt
├── solution.ipynb                 # Аналитика, EDA, запуск обучения/инференса
```

---

## Быстрый старт

### 1. Установка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Подготовка данных

- Поместите файлы `dish.csv`, `ingredients.csv` и папку `images/` в директорию `data/`
- Обновите пути в `config/effinet_multihot.yaml`, если требуется

### 3. Обучение

```python
from src.train import train, load_cfg

cfg = load_cfg("config/effinet_multihot.yaml")
results = train(cfg)
```

### 4. Предсказание

```python
from src.infer import predict_on_test

all_preds, all_targets, all_idxs, test_mae, test_ds, dish_df = predict_on_test(
    CHECKPOINT_PATH, data_dir='./data', device='cpu'
)
print(f"Test MAE: {test_mae:.4f}")
```

---

## Модальности и аугментации

- Для **изображений** — RandomResizedCrop, Flip, ColorJitter, вращения.
- Для **ингредиентов** — перетасовка, случайное удаление части тегов.

---

## Конфиг

Все параметры модели/тренировки/аугментаций — в `config/effinet_multihot.yaml`

---

## Тесты

Для проверки аугментаций:
```bash
pytest tests/test_transform.py
```

---

## Требования

- Python 3.8+
- torch, pandas, numpy, pillow

Устанавливается через:

```bash
pip install -r requirements.txt
```

---

## Автор

Boris Demkov (borisfox)

---

**PS:** Запускать обучение удобно из `solution.ipynb` — там же есть подробная EDA-подготовка.
