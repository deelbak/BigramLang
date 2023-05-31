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


def read_names_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        names = file.read().splitlines()
    return names

def compute_bigram_probabilities(names):
    bigram_counts = {}
    for name in names:
        name = '^' + name + '$'
        for i in range(len(name) - 1):
            bigram = name[i:i+2]
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    total_bigrams = sum(bigram_counts.values())
    bigram_probabilities = {bigram: count / total_bigrams for bigram, count in bigram_counts.items()}
    
    return bigram_probabilities

def generate_name(bigram_probabilities):
    name = random.choice(list(bigram_probabilities.keys()))
    while not name.endswith('$'):
        last_char = name[-1]
        eligible_bigrams = [bigram for bigram in bigram_probabilities.keys() if bigram.startswith(last_char)]
        probabilities = [bigram_probabilities[bigram] for bigram in eligible_bigrams]
        next_char = random.choices(eligible_bigrams, weights=probabilities)[0][1]
        name += next_char
    return name.replace('^', '').replace('$', '')

file_path = 'names.txt'
names = read_names_from_file(file_path)
bigram_probabilities = compute_bigram_probabilities(names)

generated_name = generate_name(bigram_probabilities)
print(generated_name)



