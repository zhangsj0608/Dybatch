import tensorflow as tf
import os, sys
from tree import BinaryTree, TreeOps


class NarryLSTMCell(object):

    def __init__(self, hidden_size, embedding_size):
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self.subfix_list = ['i', 'j', 'f', 'o']
        self._built = False
        self.W = None
        self.U0 = None
        self.U1 = None
        self.b = None

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def embedding_size(self):
        return self._embedding_size

    def build(self):
        if self._built:
            return

        self.W = tf.get_variable(name='W_', shape=[self.embedding_size, self.hidden_size * 4], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer)
        self.U0 = tf.get_variable(name='U0_', shape=[self.hidden_size, self.hidden_size * 4], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer)
        self.U1 = tf.get_variable(name='U1_', shape=[self.hidden_size, self.hidden_size * 4], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer)
        self.b = tf.get_variable(name='b_', shape=[self.hidden_size * 4], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer)
        self._built = True

    def call(self, inputs, states):
        """
        An operation to construct a lstm cell
        :param inputs: inputs of shape [batch_size, embedding_size]
        :param states: a tuple containing the h0, c0 and h1, c1
        :return: A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`)
        """
        # 四个states，状态值有顺序
        h0, c0, h1, c1 = states
        # 两个inputs, 两个输入取平均
        x0, x1 = inputs

        # self.W = tf.concat(self.W_list, axis=1)
        # self.U0 = tf.concat(self.U0_list, axis=1)
        # self.U1 = tf.concat(self.U1_list, axis=1)
        # self.b = tf.concat(self.b_list, axis=1)

        x = tf.add(x0, x1) / 2
        linear = tf.matmul(x, self.W) + tf.matmul(h0, self.U0) + tf.matmul(h1, self.U1) + self.b

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(linear, num_or_size_splits=4, axis=1)
        new_c = tf.add(tf.add(tf.multiply(tf.nn.sigmoid(f), c0), tf.multiply(tf.nn.sigmoid(f), c1)),
                       tf.multiply(tf.nn.sigmoid(i), tf.nn.tanh(j)))
        new_h = tf.multiply(tf.sigmoid(o), tf.tanh(new_c))

        return new_h, new_c


def main():
    hidden_size = 10
    batch_size = 2
    length = 10  # 动态变化的长度，一个batch中的句子的长度是相同的
    embedding_size = 100

    with tf.variable_scope('tree_lstm') as scope:
        lstm_cell = NarryLSTMCell(hidden_size=hidden_size, embedding_size=embedding_size)
        lstm_cell.build()

    with tf.Session() as tfs:
        # output1, state1 = tfs.run([output, state])
        # print(output1, state1)
        pass


if __name__ == '__main__':
    main()










