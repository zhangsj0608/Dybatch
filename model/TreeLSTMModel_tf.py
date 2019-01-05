import tensorflow as tf
import numpy as np

from rnn.TreeLSTMCell import NarryLSTMCell
from rnn.TreeLSTM import tree_lstm
from tree.BinaryTree import PPTree
from tree.TreeOps import *
from ios.Embedding import Embeddings


flags = tf.app.flags


flags.DEFINE_string('train_file', '../data/train_set.txt', 'The file path of train set')
flags.DEFINE_string('test_file', '../data/test_set.txt', 'The file path of the test file')
flags.DEFINE_string('dev_file', '../data/dev_set.txt', 'The file path of dev set')
flags.DEFINE_string('dict_file', '../data/glove.6B.100d.txt', 'The file path of words dictionary')

flags.DEFINE_integer('batch_size', 50, 'batch size')
flags.DEFINE_integer('embedding_size', 100, 'embedding size')
flags.DEFINE_integer('hidden_size', 20, 'hidden size')
flags.DEFINE_boolean('is_train', True, 'is train')
flags.DEFINE_integer('dict_size', 400001, 'size of dictionary')
flags.DEFINE_integer('full_connect_size', 50, 'size of full connect cell')
flags.DEFINE_boolean('is_shuffle', True, 'should shuffle')
flags.DEFINE_integer('num_classes', 5, 'number of classes')

FLAGS = flags.FLAGS

print('Info: reading embedding dictionary')
embeddings = Embeddings(file_name=FLAGS.dict_file, num_words=FLAGS.dict_size, dim=FLAGS.embedding_size)
print('Info: successful in reading embedding dictionary')

learning_rate = 0.01
num_epochs = 1000


def read_inputs(path):
    """
    读入数据
    :param path:
    :return: 所有样本的list，长度为3，每一行为总长度长的数据，包括sentence list, pp_tree list 和label list
    """
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


def batch_inputs(inputs, batch_size, is_shuffle):
    """
    将输入batch成不同的批，顺带根据is_shuffle来洗牌
    :param batch_size: Batch_size int
    :param is_shuffle: boolean
    :param inputs: [sentence_list, pp_tree_list, label_list], shape[3 * num_samples]
    :return: batched 样本list，每个元素是一个batch,长度为batch_size, 每一行包括：句子list, pp_tree list, label list
    """
    inputs = np.array(inputs)
    inputs = inputs.T
    num_samples = inputs.shape[0]

    batched_samples_list = []

    if is_shuffle:
        np.random.shuffle(inputs)
    i = 0
    while i * batch_size < num_samples:
        left = i * batch_size
        right = (i + 1) * batch_size if (i + 1) * batch_size < num_samples else num_samples
        batched_samples_list.append(inputs[left:right])
        i = i + 1
    return batched_samples_list


def parse_inputs(batched_samples):
    """
    对每一个batch的数据进行解析，构成同一长度的embedding_matrix和merged_tree,是lstm输入前的最后一步
    :param batched_samples: batched samples, 形状为[batch_size * 3], 每一列为sentenc list, pp_tree list和label
    :return: 样本embedding后的矩阵，形状为 [batch_size, sentence_len, embedding_size]; pp_tree list，是一个树节点列表，长度为
            2 * sentence - 1， ; label: np.ndarray, 包含float32的标签, 长度为batch_size
    """
    batch_size = len(batched_samples)
    labels = [label for _, _, label in batched_samples]  # float
    labels_arr = np.array(labels, dtype=np.int32)

    pp_trees = [None] * batch_size
    tree_leaves_encodings = [None] * batch_size
    tree_leaves_words = [None] * batch_size

    i = 0
    for sentence, pp_tree, _ in batched_samples:
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
    merged_pp_tree = generate_parent_list(merged_tree_encoding)

    # embedding 句子，将句子列表转换为矩阵，形状为[batch_size, sentence_len, embedding_size]
    to_embedding = lambda sentence: [embeddings.word2embedding(word) for word in sentence]

    words_matrix = [to_embedding(sentence) for sentence in sentence_list]

    return words_matrix, labels_arr, merged_pp_tree


def main():
    inputs = read_inputs(FLAGS.train_file)

    # 在这里搭建RNN吧

    words_pl = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, FLAGS.embedding_size], name='words')
    labels_pl = tf.placeholder(tf.int32, shape=[FLAGS.batch_size], name='labels')
    pp_tree_pl = tf.placeholder(tf.int32, shape=[None], name='parent_point_tree')

    # dataset = tf.data.Dataset.from_tensors({
    #     'words': words_pl,
    #     'pp_tree': pp_tree_pl,
    #     'labels': labels_pl
    # })
    #
    # print(dataset.output_shapes)
    # print(dataset.output_types)
    #
    # iterator = dataset.make_initializable_iterator()
    # initializer = iterator.initializer
    # next_elem = iterator.get_next()
    #
    # # input tensors
    # words_matrix = next_elem['words']
    # pp_tree = next_elem['pp_tree']
    # labels = next_elem['labels']

    with tf.variable_scope('tree_lstm') as scope:
        lstm_cell = NarryLSTMCell(FLAGS.hidden_size, FLAGS.embedding_size)
        lstm_cell.build()

        hidden, states = tree_lstm(lstm_cell, pp_tree_pl, FLAGS.batch_size, words_pl)

    with tf.variable_scope('full_connect') as scope1:
        w0 = tf.get_variable('w0', shape=[FLAGS.hidden_size, FLAGS.full_connect_size],
                             initializer=tf.random_normal_initializer)
        b0 = tf.get_variable('b0', shape=[FLAGS.full_connect_size], initializer=tf.random_normal_initializer)
        full_connect = tf.tanh(tf.matmul(hidden, w0) + b0)

    with tf.variable_scope('output') as scope1:
        w1 = tf.get_variable('w0', shape=[FLAGS.full_connect_size, FLAGS.num_classes],
                             initializer=tf.random_normal_initializer)
        b1 = tf.get_variable('b0', shape=[FLAGS.num_classes], initializer=tf.random_normal_initializer)
        y = tf.tanh(tf.matmul(full_connect, w1) + b1)

    # the optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels_pl))
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # the model accuracy, 对batch 而言的准确度
    label_pred = tf.cast(tf.argmax(y, axis=1), tf.int32)
    equal = tf.cast(tf.equal(label_pred, labels_pl), tf.float32)
    acc = tf.reduce_mean(equal)

    # train model
    with tf.Session() as tfs:
        tf.global_variables_initializer().run()

        for epoch in range(num_epochs):

            batch_loss = 0.0
            batch_acc = 0.0

            whole_batches = batch_inputs(inputs, batch_size=FLAGS.batch_size, is_shuffle=True)

            for num_batches in range(len(whole_batches)):
                    words_matrix_dt, labels_arr_dt, pp_tree_dt = parse_inputs(whole_batches[num_batches])

                    # tfs.run([initializer], feed_dict={words_pl: words_matrix_dt, labels_pl: labels_arr_dt,
                    #                                   pp_tree_pl: pp_tree_dt})
                    _, b_loss = tfs.run([optimizer, loss],
                                        feed_dict={words_pl: words_matrix_dt, labels_pl: labels_arr_dt,
                                                   pp_tree_pl: pp_tree_dt})
                    b_acc = tfs.run(acc,
                                    feed_dict={words_pl: words_matrix_dt, labels_pl: labels_arr_dt,
                                               pp_tree_pl: pp_tree_dt})

                    batch_loss = batch_loss + b_loss
                    batch_acc = batch_acc + b_acc

                    print('batch {0:4d} batch loss {1:.4f} batch accuracy {2:.4f}'.
                          format(num_batches, b_loss, b_acc))

                    num_batches = num_batches + 1

            epoch_loss = batch_loss / num_batches
            epoch_acc = batch_acc / num_batches

            print('epoch {0:4d}: loss {1: .5f}: accuracy {2: .5f}'.format(epoch, epoch_loss, epoch_acc))


if __name__ == '__main__':
    main()
