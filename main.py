import pandas as pd
import random

# Чтение данных из файла
with open('names.txt', 'r') as file:
    names = file.read().splitlines()

# Создание словаря для подсчета биграмм и их частот
bigram_counts = {}

# Вычисление вероятности всех существующих биграмм
for name in names:
    for i in range(len(name)-1):
        bigram = name[i:i+2]
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1

# Вычисление суммарной частоты биграмм
total_count = sum(bigram_counts.values())

# Вычисление вероятности для каждой биграммы
bigram_probabilities = {bigram: count/total_count for bigram, count in bigram_counts.items()}

# Визуализация таблицы вероятностей биграмм
df = pd.DataFrame({'Bigram': list(bigram_probabilities.keys()), 'Probability': list(bigram_probabilities.values())})
table = df.pivot(index='Bigram', columns='Probability').fillna(0)
print(table)

# Функция для генерации имени
def generate_name():
    name = random.choice(list(bigram_probabilities.keys()))
    while not name.endswith('$'):
        last_char = name[-1]
        eligible_bigrams = [bigram for bigram in bigram_probabilities.keys() if bigram.startswith(last_char)]
        probabilities = [bigram_probabilities[bigram] for bigram in eligible_bigrams]
        next_char = random.choices(eligible_bigrams, weights=probabilities)[0][1]
        name += next_char
    return name.replace('^', '').replace('$', '')

# Генерация имени
generated_name = generate_name()
print("Generated Name:", generated_name)
