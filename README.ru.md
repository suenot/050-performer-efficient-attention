# Глава 52: Performer — Эффективное внимание с FAVOR+

Эта глава посвящена **Performer** — архитектуре Transformer, которая достигает линейной сложности по времени и памяти благодаря механизму FAVOR+ (Fast Attention Via positive Orthogonal Random features). В отличие от стандартных Transformer с квадратичной сложностью внимания O(L²), Performer масштабируется линейно O(L), что делает его идеальным для обработки длинных финансовых временных рядов.

<p align="center">
<img src="https://i.imgur.com/8KvZmWJ.png" width="70%">
</p>

## Содержание

1. [Введение в Performer](#введение-в-performer)
    * [Узкое место внимания](#узкое-место-внимания)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими методами эффективного внимания](#сравнение-с-другими-методами-эффективного-внимания)
2. [Механизм FAVOR+](#механизм-favor)
    * [Ядерный трюк](#ядерный-трюк)
    * [Случайные признаки Фурье](#случайные-признаки-фурье)
    * [Положительные случайные признаки](#положительные-случайные-признаки)
    * [Ортогональные случайные признаки](#ортогональные-случайные-признаки)
3. [Математические основы](#математические-основы)
    * [Стандартная формулировка внимания](#стандартная-формулировка-внимания)
    * [Ядерная переформулировка](#ядерная-переформулировка)
    * [Аппроксимация картой признаков](#аппроксимация-картой-признаков)
    * [Анализ сложности](#анализ-сложности)
4. [Практические примеры](#практические-примеры)
5. [Реализация на Rust](#реализация-на-rust)
6. [Реализация на Python](#реализация-на-python)
7. [Лучшие практики](#лучшие-практики)
8. [Ресурсы](#ресурсы)

## Введение в Performer

Performer — это архитектура Transformer, представленная Google Research в 2020 году, которая решает фундаментальное ограничение квадратичной сложности стандартных Transformer. Используя FAVOR+ (Fast Attention Via positive Orthogonal Random features), Performers аппроксимируют softmax-внимание с доказуемой точностью, сохраняя линейную сложность.

### Узкое место внимания

Стандартное внимание Transformer имеет сложность O(L²), где L — длина последовательности:

```
Стандартное внимание:
┌─────────────────────────────────────────────────────┐
│  Attention(Q, K, V) = softmax(QK^T / √d) · V        │
│                                                      │
│  A = softmax(QK^T / √d)   ← Эта матрица L × L       │
│                                                      │
│  Для L = 1000:  A имеет 1 000 000 элементов         │
│  Для L = 10000: A имеет 100 000 000 элементов       │
└─────────────────────────────────────────────────────┘
```

Это становится непрактичным для длинных финансовых временных рядов:
- **Тиковые данные**: Миллионы точек данных в день
- **Мульти-активные портфели**: Необходимы длинные окна просмотра
- **Высокочастотная торговля**: Требования к обработке в реальном времени

### Ключевые преимущества

1. **Линейная сложность O(L)**
   - Масштабируется до произвольных длин последовательностей
   - Эффективен по памяти для длинных временных рядов
   - Позволяет обрабатывать тиковые данные

2. **Доказуемые гарантии аппроксимации**
   - Несмещённая оценка матрицы внимания
   - Свойства равномерной сходимости
   - Низкая дисперсия оценки

3. **Прямая замена**
   - Совместим с существующими архитектурами Transformer
   - Может использоваться с предобученными моделями
   - Тот же API, что и стандартное внимание

4. **Гибкость ядер**
   - Не ограничен softmax-вниманием
   - Может использовать другие ядерные функции
   - Позволяет создавать новые механизмы внимания

### Сравнение с другими методами эффективного внимания

| Метод | Сложность | Точный/Аппрокс | Преимущества | Недостатки |
|-------|-----------|----------------|--------------|------------|
| **Performer** | O(L) | Аппрокс | Доказуемые границы, общие ядра | Дисперсия случайных признаков |
| Linformer | O(L) | Аппрокс | Простая проекция | Фиксированная длина последовательности |
| BigBird | O(L) | Точный (разреженный) | Сохраняет точное внимание | Ручная разреженность |
| Reformer | O(L·log(L)) | Точный (LSH) | Обратимые слои | Сложная реализация |
| Flash Attention | O(L²) | Точный | IO-оптимизированный, быстрый | Всё ещё квадратичный |
| Longformer | O(L) | Точный (разреженный) | Скользящее окно | Ограниченное глобальное внимание |

## Механизм FAVOR+

FAVOR+ (Fast Attention Via positive Orthogonal Random features) — это ключевая инновация, которая обеспечивает линейное внимание в Performers.

### Ядерный трюк

Ключевое понимание заключается в том, что softmax-внимание можно рассматривать как ядерную функцию:

```python
# Стандартное внимание вычисляет:
# A[i,j] = exp(q_i · k_j / √d) / Σ_l exp(q_i · k_l / √d)

# Это эквивалентно softmax-ядру:
# K_SM(x, y) = exp(x · y)

# Если мы можем аппроксимировать это ядро картами признаков φ:
# K_SM(x, y) ≈ φ(x)^T · φ(y)

# Тогда внимание становится:
# Attention ≈ D^(-1) · φ(Q) · (φ(K)^T · V)
#                          ↑
#              Это можно вычислить за O(L·d²) вместо O(L²·d)
```

### Случайные признаки Фурье

Аппроксимация использует случайные признаки Фурье на основе теоремы Бохнера:

```
Для сдвиг-инвариантных ядер K(x, y) = K(x - y):

K(x, y) = E_ω[exp(iω^T(x - y))]
        = E_ω[cos(ω^T(x - y))] + i·E_ω[sin(ω^T(x - y))]

Карта признаков:
z(x) = [cos(ω_1^T x), sin(ω_1^T x), ..., cos(ω_m^T x), sin(ω_m^T x)]

Где ω_i ~ p(ω) (спектральная плотность ядра)
```

### Положительные случайные признаки

Стандартные случайные признаки Фурье могут давать отрицательные значения, что приводит к нестабильности обучения. FAVOR+ использует **положительные случайные признаки**:

```python
# Стандартная карта признаков (может быть отрицательной):
z_sin_cos(x) = exp(||x||²/2) · [cos(ω^T x), sin(ω^T x)]

# Положительная карта признаков (всегда положительная):
z_positive(x) = exp(-||x||²/2) · exp(ω^T x)

# Почему важны положительные признаки:
# - Веса внимания должны быть неотрицательными
# - Отрицательные аппроксимации вызывают нестабильность
# - Положительные признаки сохраняют softmax-подобное поведение
```

### Ортогональные случайные признаки

FAVOR+ дополнительно улучшает точность с помощью ортогональных случайных признаков:

```python
# IID случайные признаки:
Ω_iid = [ω_1, ω_2, ..., ω_m]  где ω_i ~ N(0, I_d)

# Ортогональные случайные признаки:
Ω_orth = Q · S  где Q ортонормальна, S диагональное масштабирование

# Преимущества ортогональности:
# - Меньшая дисперсия в аппроксимации ядра
# - Лучшее покрытие пространства признаков
# - Сохраняет несмещённость
```

## Математические основы

### Стандартная формулировка внимания

Даны запросы Q, ключи K, значения V ∈ ℝ^(L×d):

```
Attention(Q, K, V) = softmax(QK^T / √d) · V

Где softmax применяется построчно:
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Матрица внимания A ∈ ℝ^(L×L):
```
A = softmax(QK^T / √d)
A_ij = exp(q_i · k_j / √d) / Σ_l exp(q_i · k_l / √d)
```

### Ядерная переформулировка

Выразим внимание через ядерную нотацию:
```
A = D^(-1) · exp(QK^T / √d)

Где D = diag(exp(QK^T / √d) · 1_L) — нормализация
```

Softmax-ядро:
```
K_SM(q, k) = exp(q · k)

Может быть разложено:
exp(q · k) = exp(||q||²/2) · exp(-||q-k||²/2) · exp(||k||²/2)
                                ↑
                        Гауссово ядро
```

### Аппроксимация картой признаков

Карта признаков FAVOR+ φ: ℝ^d → ℝ^m:

```python
def favor_plus_feature_map(x, omega, scale=True):
    """
    Положительная ортогональная карта случайных признаков FAVOR+.

    Args:
        x: Входные векторы [batch, length, d]
        omega: Случайные признаки [m, d] (ортогонализированные)
        scale: Применять ли масштабирование d^(-1/4)

    Returns:
        Векторы признаков [batch, length, m]
    """
    if scale:
        x = x / (x.shape[-1] ** 0.25)  # масштабирование d^(-1/4)

    # Проекция на случайные признаки
    x_omega = x @ omega.T  # [batch, length, m]

    # Положительная карта признаков: exp(-||x||²/2) · exp(ω^T x)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
    phi = torch.exp(-x_norm_sq / 2 + x_omega)

    return phi / math.sqrt(omega.shape[0])  # Нормализация на √m
```

### Анализ сложности

**Стандартное внимание:**
```
1. Вычислить QK^T:           O(L² · d)
2. Применить softmax:        O(L²)
3. Умножить на V:            O(L² · d)
Итого:                       O(L² · d)
Память для A:                O(L²)
```

**Performer FAVOR+:**
```
1. Вычислить φ(Q), φ(K):     O(L · m · d)
2. Вычислить φ(K)^T · V:     O(L · m · d)  ← Ключевое понимание!
3. Вычислить φ(Q) · результат: O(L · m · d)
4. Нормализация:             O(L · m)
Итого:                       O(L · m · d)
Память:                      O(L · m + m · d)
```

Когда m << L (обычно m ~ O(d·log(d))):
- Время: O(L · d² · log(d)) против O(L² · d)
- Память: O(L · d · log(d)) против O(L²)

## Практические примеры

### 01: Подготовка данных

```python
# python/01_data_preparation.py

import pandas as pd
import numpy as np
from typing import List, Dict

def prepare_performer_data(
    symbols: List[str],
    lookback: int = 512,  # Performer справляется с длинными последовательностями
    horizon: int = 24,
    features: List[str] = ['log_return', 'volume_change', 'volatility', 'rsi']
) -> Dict:
    """
    Подготовка данных для обучения Performer.

    Линейная сложность Performer позволяет использовать более длинные
    окна просмотра по сравнению со стандартными Transformers.
    """
    all_data = []

    for symbol in symbols:
        df = load_bybit_data(symbol)

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['log_return'].rolling(20).std()
        df['rsi'] = compute_rsi(df['close'], 14)

        all_data.append(df[features + ['log_return']].dropna())

    aligned_data = pd.concat(all_data, axis=1, keys=symbols).dropna()

    X, y = [], []
    for i in range(lookback, len(aligned_data) - horizon):
        X.append(aligned_data.iloc[i-lookback:i].values)
        y.append(aligned_data[symbols[0]]['log_return'].iloc[i:i+horizon].values)

    return {'X': np.array(X), 'y': np.array(y), 'symbols': symbols}
```

### 02: Архитектура Performer

Смотрите [python/model.py](python/model.py) для полной реализации.

### 03: Обучение модели

```python
# python/03_train_model.py

config = {
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 512,
    'num_features': 64,      # Размерность случайных признаков
    'use_orthogonal': True,  # Использовать ортогональные признаки
    'dropout': 0.1,
    'max_seq_len': 1024,     # Может обрабатывать длинные последовательности!
    'prediction_horizon': 24
}

model = PerformerForecaster(**config)

for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
```

### 04: Предсказание финансовых временных рядов

```python
# python/04_prediction.py

def predict_returns(model, data, horizon=24):
    """Предсказание будущих доходностей с оценкой неопределённости."""
    model.eval()

    with torch.no_grad():
        predictions = model(data)

        # Monte Carlo dropout для оценки неопределённости
        model.train()
        mc_predictions = []
        for _ in range(100):
            with torch.no_grad():
                mc_predictions.append(model(data))

        mc_predictions = torch.stack(mc_predictions)
        mean_pred = mc_predictions.mean(dim=0)
        std_pred = mc_predictions.std(dim=0)

    return {
        'predictions': predictions.numpy(),
        'mean': mean_pred.numpy(),
        'std': std_pred.numpy(),
        'lower_95': (mean_pred - 1.96 * std_pred).numpy(),
        'upper_95': (mean_pred + 1.96 * std_pred).numpy()
    }
```

### 05: Бэктестинг стратегии

```python
# python/05_backtest.py

def backtest_performer_strategy(
    model, test_data,
    initial_capital=100000,
    transaction_cost=0.001
):
    """Бэктест торговой стратегии на основе Performer."""
    capital = initial_capital
    position = 0.0

    for batch_x, batch_y in test_data:
        with torch.no_grad():
            predictions = model(batch_x)
            pred_return = predictions[0, 0].item()
            actual_return = batch_y[0, 0].item()

            if pred_return > 0.001:
                target_position = 1.0
            elif pred_return < -0.001:
                target_position = -1.0
            else:
                target_position = 0.0

            costs = abs(target_position - position) * transaction_cost * capital
            position = target_position
            pnl = position * actual_return * capital - costs
            capital += pnl

    return {'total_return': (capital - initial_capital) / initial_capital}
```

## Реализация на Rust

Смотрите [rust_performer](rust_performer/) для полной реализации на Rust с интеграцией Bybit.

```
rust_performer/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Главный экспорт библиотеки
│   ├── api/                # Клиент Bybit API
│   ├── data/               # Обработка данных
│   ├── model/              # Архитектура Performer
│   │   ├── config.rs       # Конфигурация модели
│   │   ├── favor.rs        # Механизм внимания FAVOR+
│   │   └── performer.rs    # Полная модель
│   └── strategy/           # Торговая стратегия
└── examples/
    ├── fetch_data.rs
    ├── train.rs
    └── backtest.rs
```

### Быстрый старт (Rust)

```bash
cd rust_performer
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT
cargo run --example train -- --epochs 100 --batch-size 32
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Реализация на Python

Смотрите [python/](python/) для реализации на Python.

```
python/
├── model.py                # Реализация модели Performer
├── data.py                 # Загрузка и предобработка данных
├── strategy.py             # Торговая стратегия и бэктестинг
├── example_usage.py        # Полный пример
└── requirements.txt        # Зависимости
```

### Быстрый старт (Python)

```bash
pip install -r requirements.txt
python example_usage.py
```

## Лучшие практики

### Когда использовать Performer

**Идеальные случаи:**
- Моделирование длинных последовательностей (тиковые данные, книга ордеров)
- Ограниченные по памяти среды
- Требования к выводу в реальном времени
- Многогоризонтное прогнозирование

**Рассмотрите альтернативы для:**
- Очень коротких последовательностей (L < 100) — стандартное внимание может быть быстрее
- Задач, требующих точных паттернов внимания
- Когда важна интерпретируемость весов внимания

### Рекомендации по гиперпараметрам

| Параметр | Рекомендуется | Примечания |
|----------|---------------|------------|
| `num_features` | d·log(d) | Баланс точности и скорости |
| `use_orthogonal` | True | Лучшая точность, минимальные накладные расходы |
| `num_layers` | 4-6 | Глубже, чем стандарт для длинных последовательностей |
| `d_model` | 128-256 | Стандартные размеры Transformer работают |
| `dropout` | 0.1-0.2 | Регуляризация для стабильности |

### Типичные ошибки

1. **Слишком маленькая размерность признаков**: Используйте минимум d·log(d) признаков
2. **Неиспользование ортогональных признаков**: Ведёт к высокой дисперсии
3. **Забывание epsilon в нормализации**: Вызывает деление на ноль
4. **Использование каузального внимания для двунаправленных задач**: Лишняя сложность

### Сравнение памяти

Для длины последовательности L = 4096, d = 256:

| Метод | Память внимания | Общая память |
|-------|-----------------|--------------|
| Стандартный | 64 МБ | ~100 МБ |
| Performer | 4 МБ | ~40 МБ |
| Экономия | 16x | 2.5x |

## Ресурсы

### Статьи

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Оригинальная статья FAVOR+ (2020)
- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) — Основа случайных признаков Фурье
- [Orthogonal Random Features](https://arxiv.org/abs/1610.09072) — Улучшения ортогональных признаков
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Теоретические инсайты

### Реализации

- [Google Research Performer](https://github.com/google-research/google-research/tree/master/performer) — Официальная реализация
- [Hugging Face Performer](https://huggingface.co/docs/transformers/model_doc/performer) — Интеграция HuggingFace
- [Phil Wang's Performer PyTorch](https://github.com/lucidrains/performer-pytorch) — Чистая реализация PyTorch

### Связанные главы

- [Глава 51: Linformer для длинных последовательностей](../51_linformer_long_sequences) — Подход линейной проекции
- [Глава 53: BigBird разреженное внимание](../53_bigbird_sparse_attention) — Паттерны разреженного внимания
- [Глава 54: Reformer LSH внимание](../54_reformer_lsh_attention) — Locality-sensitive hashing
- [Глава 58: Flash Attention для трейдинга](../58_flash_attention_trading) — IO-оптимизированное точное внимание

---

## Уровень сложности

**Средний и продвинутый**

Предварительные требования:
- Основы архитектуры Transformer
- Линейная алгебра (матричная декомпозиция, ядерные методы)
- Теория вероятностей (случайные признаки, границы концентрации)
- Программирование PyTorch/Rust ML
