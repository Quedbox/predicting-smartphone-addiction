# Predicting Smartphone Addiction — OOF Stacking Project

Machine Learning / Ensemble Learning проект по ранжированию пользователей по вероятности `addicted_label` на данных Kaggle Playground Series S6E8.

**Результат: 216 / 3520 — Top ~6% leaderboard.**

Основная техническая часть проекта — построение многоуровневого OOF-stacking pipeline: объединение predictions разных моделей, контроль alignment, фильтрация почти дублирующихся сигналов, Rank-Gauss преобразование и cross-fitted Logistic Regression в качестве meta-model.

## Задача

Competition data описывает пользователей набором поведенческих и других признаков. Необходимо предсказать вероятность положительного класса `addicted_label`.

С продуктовой точки зрения такую постановку можно интерпретировать как построение **risk score**, который ранжирует пользователей по вероятности проблемного паттерна использования смартфона.

В гипотетическом digital-wellbeing продукте такой score мог бы использоваться как входной сигнал для:

- приоритизации пользователей для добровольных wellbeing-механик;
- персонализации рекомендаций;
- определения групп, для которых стоит дополнительно анализировать продуктовые паттерны поведения;
- offline-оценки эффективности разных моделей ранжирования.

Данные соревнования синтетические, поэтому этот проект не является медицинской диагностической системой.

## Метрика

Основная метрика — **ROC-AUC**.

Это означает, что модель оценивается по качеству ранжирования: положительные объекты должны в среднем получать более высокий score, чем отрицательные. Фиксированный classification threshold для competition result не требуется.

## Подход

Вместо обучения одной финальной модели решение использует stacking.

```text
public OOF/test predictions
        ↓
shape / alignment / finite-value checks
        ↓
individual OOF ROC-AUC
        ↓
near-duplicate filtering
        ↓
self-referential member filtering
        ↓
percentile ranks
        ↓
Rank-Gauss
        ↓
cross-fitted Logistic Regression
        ↓
validated stack
        ↓
competition-only test blend
```

### OOF library

Каждая базовая модель представлена двумя массивами:

- OOF predictions для competition train;
- predictions той же модели для competition test.

OOF predictions позволяют использовать predictions базовых моделей как признаки второго уровня без прямой утечки target.

В проекте объединяются публично доступные Kaggle OOF libraries разных моделей и ансамблей. Перед stacking проверяются размеры массивов и наличие нечисловых значений.

### Diversity filtering

Большое количество моделей само по себе не гарантирует сильный ensemble.

Predictions переводятся в percentile ranks, после чего вычисляется rank-correlation. Если две модели имеют correlation выше `0.9995`, одна из них удаляется как практически дублирующий сигнал.

Дополнительно исключаются известные self-referential members — predictions, которые сами уже являются stacks/blends поверх части используемой библиотеки.

### Rank-Gauss

Базовые модели могут иметь разную калибровку probabilities. Поскольку ROC-AUC зависит прежде всего от порядка объектов, predictions сначала переводятся в percentile ranks, а затем через inverse normal CDF — в Rank-Gauss features.

Так разные модели попадают в сопоставимый feature space, сохраняя своё ранжирование.

### Meta-model

Второй уровень — Logistic Regression.

Метамодель не пытается заново моделировать исходные признаки. Её задача — выучить устойчивую комбинацию уже сильных model predictions.

Для оценки используется cross-fitting на фиксированных `StratifiedKFold`: каждая строка получает prediction метамодели, которая не обучалась на этой строке.

После OOF-оценки Logistic Regression обучается на полной meta-feature matrix и формирует test prediction.

### Competition blend

Чистый stack сохраняется отдельно.

Финальный Kaggle submission дополнительно смешивает stack с сильным публичным test prediction. Этот этап отделён от основной модели, потому что test-only public prediction невозможно валидировать через ту же OOF-схему.

## Результат

Итоговый leaderboard result:

**216 место из 3520 — Top ~6%.**

Проект для меня стал практикой в:

- OOF predictions и cross-fitting;
- stacking;
- ROC-AUC oriented modeling;
- анализе корреляции predictions;
- ensemble diversity;
- rank averaging;
- Rank-Gauss transformations;
- meta-modeling;
- защите от leakage и misalignment;
- воспроизводимой работе с большим количеством model artifacts.

## Что именно сделано мной

Этот пункт важен для корректной интерпретации проекта.

Я не заявляю авторство всех базовых моделей, predictions которых используются в OOF library. Часть OOF/test artifacts была опубликована участниками Kaggle и используется как исходный набор meta-features.

Моя работа в рамках этого решения:

- сбор и унификация нескольких OOF/test libraries;
- проверки размеров, порядка и корректности predictions;
- расчёт индивидуального OOF ROC-AUC;
- фильтрация почти идентичных predictors;
- удаление self-referential ensemble members;
- percentile-rank и Rank-Gauss transformations;
- cross-fitted обучение Logistic Regression второго уровня;
- локальная OOF-оценка stack;
- построение test predictions;
- отделение валидируемого stack от competition-only blend;
- подготовка воспроизводимого notebook и структуры проекта.

## Структура репозитория

```text
Predicting-Smartphone-Addiction/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── predicting-smartphone-addiction.ipynb
│
└── assets/
    └── leaderboard.png
```

`assets/leaderboard.png` необязателен, но полезен для GitHub-версии проекта: туда можно положить скриншот финального места.

## Данные

Competition data не хранится в repository.

Исходные `train.csv`, `test.csv` и `sample_submission.csv` доступны на странице данных Kaggle competition:

https://www.kaggle.com/competitions/playground-series-s6e8/data

Для полного воспроизведения stacking pipeline также нужны публичные OOF/test datasets, перечисленные в `data/README.md`.

Самый простой способ воспроизведения — запуск notebook в Kaggle с подключёнными competition data и необходимыми public datasets.

## Локальный запуск

Проект использует `uv` для управления Python environment и зависимостями.

```bash
uv sync
```

После установки окружения откройте:

```text
notebooks/predicting-smartphone-addiction.ipynb
```

в VS Code/Jupyter.

Notebook по умолчанию ищет входные данные в:

```text
/kaggle/input
```

Для локального зеркала Kaggle inputs можно задать переменную окружения:

```text
S6E8_INPUT=/path/to/s6e8-inputs
```

Подробная структура входных данных описана в `data/README.md`.

## Технологии

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Jupyter
- uv

Ключевые ML-концепции:

- binary classification;
- ROC-AUC;
- Stratified K-Fold;
- OOF predictions;
- stacking;
- rank-based ensembling;
- Rank-Gauss;
- Logistic Regression meta-model;
- cross-fitting.

## Ограничения

Проект построен на synthetic competition dataset и не предназначен для реального определения или медицинской диагностики зависимости от смартфона.

Кроме того, итоговый competition blend включает public test predictions и поэтому не имеет полностью эквивалентной локальной OOF-оценки. По этой причине notebook отдельно сохраняет prediction чистой cross-fitted meta-model.
