import random
import math
import json


class NN:
    def __init__(self, layer_nodes):
        self.layer_nodes = layer_nodes
        self.weights = []
        self.biases = []
        for i in range(1, len(self.layer_nodes)):
            weight_layer = []
            for _ in range(self.layer_nodes[i]):
                weights_for_node = []
                for _ in range(self.layer_nodes[i - 1]):
                    weights_for_node.append(random.random())
                weight_layer.append(weights_for_node)
            self.weights.append(weight_layer)

        for i in range(1, len(self.layer_nodes)):
            layer_biases = []
            for _ in range(self.layer_nodes[i]):
                layer_biases.append(random.random())
            self.biases.append(layer_biases)

    def run(self, input_):
        prev = input_
        for i in range(1, len(self.layer_nodes)):
            cur_layer = []
            for j in range(self.layer_nodes[i]):
                summation = 0
                for k in range(self.layer_nodes[i - 1]):
                    summation += prev[k] * self.weights[i - 1][j][k]
                result = self.activation_function(summation + self.biases[i - 1][j])
                cur_layer.append(result)
            prev = cur_layer

        return cur_layer.index(max(cur_layer))

    def activation_function(self, x):
        return 1 / (1 + math.e**-x)

    def calculate_loss(self, sequences, labels):
        correct = 0

        for i, sequence in enumerate(sequences):
            if self.run(sequence) == labels[i]:
                correct += 1

        return correct / len(labels)

    def train(self):
        pass


with open("sarcasm.json", "r") as f:
    datastore = json.load(f)


processed_headlines = []
labels = []
for obj in datastore:
    labels.append(obj["is_sarcastic"])
    processed_headline = []
    for word in obj["headline"].split():
        processed_word = ""
        for c in word:
            if c in "abcdefghijklmnopqrstuvwxyz":
                processed_word += c
        processed_word = processed_word.lower()
        processed_headline.append(processed_word)
    processed_headlines.append(processed_headline)

word_index = {}
cur_token = 1
for h in processed_headlines:
    for word in h:
        if word not in word_index:
            word_index[word] = cur_token
            cur_token += 1

sequences = []
for h in processed_headlines:
    sequence = []
    for word in h:
        sequence.append(word_index[word])
    sequence = sequence[:31]
    while len(sequence) < 30:
        sequence.append(0)
    sequences.append(sequence)

nn = NN([30, 8, 8, 2])
print(nn.calculate_loss(sequences, labels))
