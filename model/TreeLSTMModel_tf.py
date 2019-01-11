import tensorflow as tf
import numpy as np
import os

from rnn.TreeLSTMCell import NarryLSTMCell
from rnn.TreeLSTM import tree_lstm
from tree.BinaryTree import PPTree
from tree.TreeOps import *
from ios.Embedding import Embeddings

import functools
from tensorflow.python.client import timeline
import time

flags = tf.app.flags


flags.DEFINE_string('train_file', '../data/train_set.txt', 'The file path of train set')
flags.DEFINE_string('test_file', '../data/test_set.txt', 'The file path of the test file')
flags.DEFINE_string('dev_file', '../data/dev_set.txt', 'The file path of dev set')
flags.DEFINE_string('dict_file', '../data/glove.6B.50d.txt', 'The file path of words dictionary')

flags.DEFINE_integer('batch_size', 20, 'batch size')
flags.DEFINE_integer('embedding_size', 50, 'embedding size')
flags.DEFINE_integer('hidden_size', 150, 'hidden size')
flags.DEFINE_boolean('is_train', True, 'is train')
flags.DEFINE_integer('dict_size', 400001, 'size of dictionary')  #
flags.DEFINE_integer('full_connect_size', 50, 'size of full connect cell')
flags.DEFINE_boolean('is_shuffle', True, 'should shuffle')
flags.DEFINE_integer('num_classes', 5, 'number of classes')

FLAGS = flags.FLAGS

learning_rate = 0.01
num_epochs = 10


def _read_inputs(path):
    """
    读入数据
    :param path:
    :return: 所有样本的list，长度为3，每一行为总长度长的数据，包括sentence list, pp_tree list 和label list
    """
    localtime = time.asctime(time.localtime(time.time()))
    print('Info: {0:} read inputs files'.format(localtime))
    sentence_list = []
    pp_tree_list = []
    label_list = []
    with open(path) as file:
        for line in file:
            sentence, pp_tree, label = line.split('|')

            words_list = sentence.split(' ')
            sentence_list.append(words_list)

            a_pp_tree = [int(num) for num in pp_tree.split(' ')]
            pp_tree_list.append(a_pp_tree)

            label_list.append(float(label))
        # label 分成5类
        label_list = [int(i / 2.0) if i < 10 else 4 for i in label_list]

    return sentence_list, pp_tree_list, label_list


def _batch_inputs(inputs, batch_size, is_shuffle):
    """
    将输入batch成不同的批，顺带根据is_shuffle来洗牌
    :param batch_size: Batch_size int
    :param is_shuffle: boolean
    :param inputs: [sentence_list, pp_tree_list, label_list], shape[3 * num_samples]
    :return: batched 样本list，每个元素是一个batch,长度为batch_size, 每一行包括：句子list, pp_tree list, label list
    """
    localtime = time.asctime(time.localtime(time.time()))
    print('Info: {0:} batch inputs samples'.format(localtime))
    inputs = np.array(inputs)
    inputs = inputs.T
    num_samples = inputs.shape[0]

    batched_samples_list = []

    if is_shuffle:
        np.random.shuffle(inputs)
    i = 0
    while (i + 1) * batch_size < num_samples:
        left = i * batch_size
        right = (i + 1) * batch_size  # if (i + 1) * batch_size < num_samples else num_samples
        batched_samples_list.append(inputs[left:right])
        i = i + 1
    return batched_samples_list


def _parse_batch(a_batch, sentence_to_matrix_func):
    """
    对每一个batch的数据进行解析，构成同一长度的embedding_matrix和merged_tree,是lstm输入前的最后一步
    """
    sentence_list = [sample[0] for sample in a_batch]
    pp_tree_list = [sample[1] for sample in a_batch]
    label_list = [sample[2] for sample in a_batch]

    batch_size = len(sentence_list)

    label_arr = np.array(label_list)

    pp_trees = [None] * batch_size
    tree_leaves_encodings = [None] * batch_size
    tree_leaves_words = [None] * batch_size

    i = 0
    for sentence, pp_tree in zip(sentence_list, pp_tree_list):
        tree = PPTree(pp_list=pp_tree, word_list=sentence)
        tree.construct_tree()
        tree.traverse_with_lrchange()  # 左右树枝交换
        pp_trees[i] = tree
        tree_leaves_encodings[i] = tree.leaves_encodings  # 交换后的编码'0100110'
        tree_leaves_words[i] = tree.leaves_str  # 交换后的句子
        i = i + 1

    # 利用treeops中的操作，构建合并后的树叶编码列表；
    # 而后用合并的编码列表和原列表、原单词生成扩展后的句子，用'-NAN-'扩展，使句子长度相等；
    # 最后利用输液编码列表，生成合并后的pp_tree
    merged_tree_encoding = construct_merged_tree(tree_leaves_encodings)
    sentence_list = extend_tree_encodings(tree_leaves_encodings, tree_leaves_words, merged_tree_encoding)
    pp_tree = np.array(generate_parent_list(merged_tree_encoding))

    # embedding 句子，将句子列表转换为矩阵，形状为[batch_size, sentence_len, embedding_size]
    sentence_matrix_arr = np.array([sentence_to_matrix_func(sentence) for sentence in sentence_list])

    return sentence_matrix_arr, label_arr, pp_tree


def _sentence_to_matrix(sentence, embeddings):

    matrix = np.zeros([len(sentence), FLAGS.embedding_size])
    for i in range(len(sentence)):
        matrix[i] = embeddings.word2embedding(sentence[i])
    return matrix


def gen_epoch():
    print('Info: reading inputs')
    inputs = _read_inputs(FLAGS.train_file)

    for i in range(num_epochs):
        a_epoch = _batch_inputs(inputs, FLAGS.batch_size, True)
        yield a_epoch


def gen_batch(epoch_gen):
    print('Info: reading embedding dictionary')
    embeddings = Embeddings(file_name=FLAGS.dict_file, num_words=FLAGS.dict_size, dim=FLAGS.embedding_size)
    _sentence_to_matrix_V2 = functools.partial(_sentence_to_matrix, embeddings=embeddings)
    for an_epoch in epoch_gen:
        for a_batch in an_epoch:
            parsed_batch = _parse_batch(a_batch, _sentence_to_matrix_V2)
            yield parsed_batch


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    epochs_gen = gen_epoch()
    batches_gen = functools.partial(gen_batch, epochs_gen)

    dataset = tf.data.Dataset.from_generator(batches_gen, output_types=(tf.float32, tf.float32, tf.float32),
                                             output_shapes=((FLAGS.batch_size, None, FLAGS.embedding_size),
                                                            (FLAGS.batch_size, ), (None, ))
                                             )
    dataset = dataset.prefetch(buffer_size=2)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    # 在这里搭建RNN吧
    with tf.device('/gpu:0'):
        with tf.variable_scope('tree_lstm') as scope:
            lstm_cell = NarryLSTMCell(FLAGS.hidden_size, FLAGS.embedding_size)
            lstm_cell.build()

            hidden, states = tree_lstm(lstm_cell, next_elem[2], FLAGS.batch_size, next_elem[0])

        with tf.variable_scope('output') as scope1:
            w0 = tf.get_variable('w0', shape=[FLAGS.hidden_size, FLAGS.num_classes],
                                 initializer=tf.random_normal_initializer)
            b0 = tf.get_variable('b0', shape=[FLAGS.num_classes], initializer=tf.random_normal_initializer)
            y = tf.tanh(tf.matmul(hidden, w0) + b0)


        # with tf.variable_scope('full_connect') as scope1:
        #     w0 = tf.get_variable('w0', shape=[FLAGS.hidden_size, FLAGS.full_connect_size],
        #                          initializer=tf.random_normal_initializer)
        #     b0 = tf.get_variable('b0', shape=[FLAGS.full_connect_size], initializer=tf.random_normal_initializer)
        #     full_connect = tf.tanh(tf.matmul(hidden, w0) + b0)
        #
        # with tf.variable_scope('output') as scope1:
        #     w1 = tf.get_variable('w0', shape=[FLAGS.full_connect_size, FLAGS.num_classes],
        #                          initializer=tf.random_normal_initializer)
        #     b1 = tf.get_variable('b0', shape=[FLAGS.num_classes], initializer=tf.random_normal_initializer)
        #     y = tf.tanh(tf.matmul(full_connect, w1) + b1)

        # the optimizer
        labels_ = tf.cast(next_elem[1], tf.int32)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels_))
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        # tf.summary.scalar('loss', loss)

        # the model accuracy, 对batch 而言的准确度
        label_pred = tf.cast(tf.argmax(y, axis=1), tf.int32)
        equal = tf.cast(tf.equal(label_pred, labels_), tf.float32)
        acc = tf.reduce_mean(equal)

        # merged = tf.summary.merge_all()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # train model
    with tf.Session(config=config) as tfs:

        train_writer = tf.summary.FileWriter('../summary/train', tfs.graph)
        tf.global_variables_initializer().run()

        for epoch in range(num_epochs):

            batch_loss = 0.0
            batch_acc = 0.0
            batch_num = 0
            while True:

                try:
                    localtime = time.asctime(time.localtime(time.time()))
                    print('Info: {0:} start training batches {1:}'.format(localtime, batch_num))
                    _, b_loss, b_acc = tfs.run([optimizer, loss, acc],
                                               options=run_options,
                                               run_metadata=run_metadata)

                    localtime = time.asctime(time.localtime(time.time()))
                    print('Info: {0:} batch{1:4d} batch loss {2:.4f} batch accuracy {3:.4f}'.
                          format(localtime, batch_num, b_loss, b_acc))

                    train_writer.add_run_metadata(run_metadata, 'step%d' % batch_num)
                    # train_writer.add_summary(summary, num_batches)

                    # t1 = timeline.Timeline(run_metadata.step_stats)
                    # ctf = t1.generate_chrome_trace_format()
                    # with open('../summary/timeline.json', 'a') as f:
                    #     f.write(ctf)

                    batch_loss = batch_loss + b_loss
                    batch_acc = batch_acc + b_acc
                    batch_num = batch_num + 1
                except tf.errors.OutOfRangeError:
                    break
            # epoch_loss = batch_loss / num_batches
            # epoch_acc = batch_acc / num_batches
            # localtime = time.asctime(time.localtime(time.time()))
            # print('Info: {0:} epoch {1:4d}: loss {2:.5f}: accuracy {3:.5f}'.format(localtime, epoch, epoch_loss,
            #                                                                        epoch_acc))


def main2():
    epochs_gen = gen_epoch()
    batch_gen = gen_batch(epochs_gen)

    for i in range(5):
        sentence_matrix_arr, label_arr, pp_tree = next(batch_gen)
        print(sentence_matrix_arr.shape, label_arr.shape, pp_tree.shape)


if __name__ == '__main__':
    main()
