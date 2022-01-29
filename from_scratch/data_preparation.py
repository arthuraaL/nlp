from typing import List
import numpy as np
import re


def read_data(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+')
    return np.array(pattern.findall(text.lower()))

def one_hot_encoding(vocab_size, word_id):
    # n_columns = len(context_words)
    # one_hot_center = np.zeros((vocab_size, 1))
    # one_hot_center[center_word] = 1
    # one_hot_contexts = np.zeros(vocab_size)
    # one_hot_contexts[context_words] = 1
    one_hot_word = np.zeros(vocab_size)
    one_hot_word[word_id] = 1
    return one_hot_word

def generate_vectors(tokens: List[str], windows_size: int):
    m = windows_size
    # Unique words in the corpus
    #unique_words = list(dict.fromkeys(tokens))
    vocab = set(tokens)
    vocab_size = len(vocab)
    word_pairs = []
    X, y = [], []
    
    # Mapping word to index
    word_to_id = {word: id for id, word in enumerate(vocab)}

    for i in range(vocab_size+1):
        center_id = i
        context_left_id = list(range(max(0, i-m), i))
        context_right_id = list(range(i+1, min(vocab_size+1, i+1+m)))
        word_pairs.append((tokens[center_id], tokens[context_left_id + context_right_id]))


    for center_word, context_words in word_pairs:
        target_word_vec = one_hot_encoding(vocab_size, word_to_id[center_word])
        context_words_vec = [one_hot_encoding(vocab_size, word_to_id[c]) for c in context_words]
        # Transform the one-hot vectors for the context words into a single vector by taking an average
        X.append(np.mean(context_words_vec, axis=0))
        y.append(target_word_vec)
    
    return np.array(X), np.array(y)

