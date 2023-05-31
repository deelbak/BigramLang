import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random



class NameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

# def read_names_from_file(file_path):
#     with open(file_path, 'r') as file:
#         names = file.readlines()
#         names = [name.strip() for name in names]
#     return names


def generate_name(model, start_letter, max_length=20):
    with torch.no_grad():
        input_tensor = torch.tensor([[start_letter]], dtype=torch.long)
        hidden = None
        name = []
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            output = output.squeeze().numpy()
            next_letter = np.random.choice(len(output), p=output)
            name.append(next_letter)
            input_tensor.fill_(next_letter)
        return ''.join([chr(letter + ord('a')) for letter in name])
def read_names_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        names = file.read().splitlines()
    return names

def calculate_bigram_probabilities(names):
    bigram_counts = {}
    total_bigrams = 0

    for name in names:
        for i in range(len(name) - 1):
            bigram = name[i:i+2]
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            total_bigrams += 1
    
    bigram_probabilities = {}
    for bigram, count in bigram_counts.items():
        probability = count / total_bigrams
        bigram_probabilities[bigram] = probability
    
    return bigram_probabilities

# Пример использования
names = read_names_from_file('names.txt')
bigram_probabilities = calculate_bigram_probabilities(names)



# Чтение имен из файла
file_path = 'names.txt'
names = read_names_from_file(file_path)

# Создание словаря символов
all_letters = ''.join(names).lower()
vocab = sorted(set(all_letters))
vocab_size = len(vocab)
char_to_index = {char: index for index, char in enumerate(vocab)}

# Преобразование имен в числовые последовательности
name_sequences = []
for name in names:
    name = name.lower()
    sequence = [char_to_index[char] for char in name]
    name_sequences.append(sequence)

# Создание и обучение модели
hidden_size = 128
model = NameGenerator(vocab_size, hidden_size, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    loss_avg = 0
    for sequence in name_sequences:
        sequence = torch.tensor(sequence, dtype=torch.long).view(-1, 1)
        target = sequence[1:]
        input_tensor = sequence[:-1]
        
        optimizer.zero_grad()
        output, _ = model(input_tensor)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
    loss_avg /= len(name_sequences)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss_avg}')

# Генерация имени
start_letter = random.choice(range(vocab_size))
generated_name = generate_name(model, start_letter)

# Вывод результатов
print(f'Сгенерированное имя: {generated_name}')
