import data
import numpy as np


class Embeddings(object):

    def __init__(self, file_name='glove.6B.100d.txt', num_words=None, dim=None):
        self.path = '../data/' + file_name
        self.word_count = num_words
        self.embedding_matrix = np.zeros(shape=(num_words, dim))
        self.word_to_id = dict()
        self.id_to_word = [None] * num_words
        self._read_embeddings()

    def _read_embeddings(self):
        """
        read file of pretrained word embedding vectors
        """

        with open(self.path) as file:
            idx = 0
            for line in file:
                words = line[:-1].split(" ")
                key_word = words[0]
                vec = np.array(words[1:]).astype(np.float)
                if key_word not in self.word_to_id:  # 词汇不在字典中
                    self.word_to_id[key_word] = idx  # 词汇转id
                    self.id_to_word[idx] = key_word  # id转词汇
                    self.embedding_matrix[idx] = vec  # 矩阵
                    idx += 1
                else:
                    continue


if __name__ == '__main__':

    fileName = 'glove.6B.100d.txt'
    word_count = 400000
    embeddings = Embeddings(file_name=fileName, num_words=word_count, dim=100)
    # embeddings.read_embeddings()
    str1 = 'house'
    id1 = embeddings.word_to_id[str1]
    vec = embeddings.embedding_matrix[id1]
    print('word: {0:} id: {1:} vec: {2:}'.format(str1, id1, vec))
