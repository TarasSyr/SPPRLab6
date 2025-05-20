import pandas as pd
import numpy as np

# Генерація даних
np.random.seed(42)
n_samples = 500  # Кількість записів

# Параметри
data = {
    'Товщина (мм)': np.round(np.random.uniform(1, 10, n_samples), 1),
    'Густина (г/см³)': np.round(np.random.uniform(2, 8, n_samples), 1),
    'Температура плавлення (°C)': np.random.randint(500, 2000, n_samples),
    'Міцність на розрив (МПа)': np.random.randint(200, 1000, n_samples),
}

# Цільова змінна (з логікою + 5% шуму)
df = pd.DataFrame(data)
df['Рекомендація'] = np.where(
    (df['Температура плавлення (°C)'] > 1500) & (df['Міцність на розрив (МПа)'] > 700), 2,
    np.where(
        (df['Температура плавлення (°C)'] > 1000) | (df['Міцність на розрив (МПа)'] > 500), 1, 0
    )
)

# Додаємо 5% шуму
noise_mask = np.random.rand(n_samples) < 0.05
df.loc[noise_mask, 'Рекомендація'] = np.random.randint(0, 3, size=noise_mask.sum())

# Зберігаємо у CSV та Excel
df.to_csv(r'../files/aviation_materials.csv', index=False, encoding='utf-8-sig')
df.to_excel(r'../files/aviation_materials.xlsx', index=False)

print(df.head())