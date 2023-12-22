'''
В задании используется база курса акций Лукойла.

Обучите простую полносвязную сеть для прогнозирования временного ряда (только close) и визуализируйте результат.

Обучите такую же архитектуру сети на прогнозирование на 10 шагов вперёд прямым способом и визуализируйте результат.

Постройте графики сравнения предсказания с оригинальным сигналом по всем 10 шагам предсказания (10 графиков на разных отдельных осях).

Сделайте те же задания с другой сетью, которая будет использовать Conv1D или LSTM слои.
'''

import gdown
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import torch

# Загрузка данных с проверкой

url1 = 'https://storage.yandexcloud.net/aiueducation/Content/base/l11/16_17.csv'
url2 = 'https://storage.yandexcloud.net/aiueducation/Content/base/l11/18_19.csv'

try:
    gdown.download(url1, quiet=True)
    gdown.download(url2, quiet=True)
    print("Файлы успешно загружены")
except Exception as e:
    print(f"Ошибка при загрузке файлов: {e}")

# Загрузка датасетов с удалением ненужных столбцов по дате и времени

data16_17 = pd.read_csv('16_17.csv', sep=';').drop(columns=['DATE', 'TIME'])
data18_19 = pd.read_csv('18_19.csv', sep=';').drop(columns=['DATE', 'TIME'])
data = pd.concat([data16_17, data18_19]).to_numpy()
print(data16_17.shape)
print(data18_19.shape)
print(data.shape)

# Отображение исходных данных от точки start и длинной length
channel_names = ['Open', 'Max', 'Min', 'Close', 'Volume']
start = 100
length = 300

fig, ax = plt.subplots(figsize=(22, 13), sharex=True)
for chn in range(4):
    ax.plot(data[start:start + length, chn],
             label=channel_names[chn])
ax.set_ylabel('Цена, руб')
ax.legend()
plt.xlabel('Время')
plt.xlim(0, length)
plt.tight_layout()
plt.show()

# Перевод в формат torch и сохранение датасета через pickle
data = torch.from_numpy(data)
print(data.shape)
with open('DataTorch.pkl', "wb") as f:
    pkl.dump(data, f)