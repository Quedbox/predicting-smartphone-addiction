# Data

Данные и большие prediction artifacts намеренно не хранятся в Git repository.

## Competition data

Основные данные находятся на странице Kaggle:

https://www.kaggle.com/competitions/playground-series-s6e8/data

Необходимые файлы:

```text
train.csv
test.csv
sample_submission.csv
```

`train.csv` содержит target:

```text
addicted_label
```

Competition metric — ROC-AUC.

## External OOF/test artifacts

Финальное решение является stacking-пайплайном, поэтому для полного воспроизведения нужны публичные Kaggle datasets с OOF/test predictions.

Текущий notebook ожидает следующие директории:

```text
s6e8-oof-library-47-models
s6e8-oof-library-11-members
s6e8-mask-augmented-oof-library
s6e8-full-best-blend-npy
s6e8-adarsh-oof-library
s6e8-golem-oof-library
s6e8-fm-lattice-blend-members
s6e8-150-fusion-local-members
s6e8-catstrall-member
s6e8-catstr-aug16

s6e8-oof-prediction-library
s6e8-50-weakest-oof-models
```

Финальный competition-only blend также использует публичный dataset `Predicting Smartphone Addiction Vault`; текущий notebook находит его по подстроке `vault`.

Все эти источники использовались как публично доступные Kaggle artifacts. Самый простой способ воспроизвести решение — найти datasets на Kaggle по указанным названиям и подключить их как Inputs к notebook.

## Recommended Kaggle setup

В Kaggle notebook подключите:

1. competition `Predicting Smartphone Addiction`;
2. перечисленные выше OOF datasets;
3. `Predicting Smartphone Addiction Vault`.

После подключения Kaggle автоматически монтирует inputs под `/kaggle/input`, откуда notebook ищет нужные файлы.

## Local setup

Для локального запуска удобно создать отдельную директорию, которая имитирует структуру Kaggle inputs:

```text
s6e8-inputs/
├── playground-series-s6e8/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── s6e8-oof-library-47-models/
├── s6e8-oof-library-11-members/
├── s6e8-mask-augmented-oof-library/
├── s6e8-full-best-blend-npy/
├── s6e8-adarsh-oof-library/
├── s6e8-golem-oof-library/
├── s6e8-fm-lattice-blend-members/
├── s6e8-150-fusion-local-members/
├── s6e8-catstrall-member/
├── s6e8-catstr-aug16/
├── s6e8-oof-prediction-library/
├── s6e8-50-weakest-oof-models/
└── ...vault.../
```

После этого укажите корень через:

```text
S6E8_INPUT=/absolute/path/to/s6e8-inputs
```

Notebook рекурсивно ищет нужные prediction artifacts внутри этой директории.

## Why the files are not committed

Причины не включать data и OOF arrays в Git:

- они значительно увеличивают размер repository;
- исходные competition data доступны напрямую через Kaggle;
- OOF libraries принадлежат их авторам и используются как публичные competition artifacts;
- repository должен содержать код, документацию и воспроизводимую структуру проекта, а не копии внешних datasets.

## Important alignment note

Большинство OOF arrays не содержит `id`.

Поэтому соответствие между:

```text
train row i
```

и:

```text
OOF prediction i
```

определяется позицией строки.

Не сортируйте и не переиндексируйте competition train перед объединением OOF arrays.
