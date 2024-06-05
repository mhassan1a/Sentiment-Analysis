import torch
import gensim
import numpy as np
import urllib.request

def download_word2vec(path):
    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    urllib.request.urlretrieve(url, path)

def load_word2vec(path):
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

def get_word2vec_embeddings(word2vec, word2idx, embedding_dim):
    embeddings = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in word2vec:
            embeddings[idx] = word2vec[word]
    return torch.from_numpy(embeddings).float()


if __name__ == "__main__":
    download_word2vec("GoogleNews-vectors-negative300.bin.gz")
    word2vec = load_word2vec("GoogleNews-vectors-negative300.bin.gz")
    print("Word2Vec loaded")
    word2idx = {"hello": 0, "world": 1}
    embeddings = get_word2vec_embeddings(word2vec, word2idx, 300)
    print(embeddings)
    print(embeddings.shape)
 
    