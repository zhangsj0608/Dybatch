import tensorflow as tf

from rnn.TreeLSTMCell import NarryLSTMCell


def tree_lstm(narry_lstm_cell, parent_idx, batch_size, words):
    """
    构造n-array lstm tree树
    :param narry_lstm_cell:  lstm cell
    :param parent_idx: 父节点列表的Tensor, 形状[total_len], 其中total_len = 2 * time_steps - 1
    :param words: 句子对应的单词的embedding的tensor, 形状[batch_size, time_steps, embedding_size]
    :return:
    """
    words = tf.transpose(words, perm=[1, 0, 2])
    total_len = tf.shape(parent_idx)[0]  # 17
    time_steps = tf.shape(words)[0]  # 9
    # batch_size = tf.shape(words)[1]
    embedding_size = tf.shape(words)[2]
    hidden_size = narry_lstm_cell.hidden_size

    parent_idx_ta = tf.TensorArray(tf.float32, size=total_len, clear_after_read=False)
    parent_idx_ta = parent_idx_ta.unstack(parent_idx)  # 父节点对应的tensor的列表

    words_ta = tf.TensorArray(tf.float32, size=time_steps, clear_after_read=False)
    words_ta = words_ta.unstack(words)

    states_ta = tf.TensorArray(tf.float32, size=time_steps - 1, element_shape=[batch_size, hidden_size * 2],
                               clear_after_read=False)

    available = tf.concat([tf.ones([time_steps], tf.float32), tf.zeros([total_len - time_steps], tf.float32)], 0)
    visited = tf.zeros([total_len], tf.float32)
    min_child_ta = tf.TensorArray(tf.float32, size=total_len, element_shape=[], clear_after_read=False)

    # 初始化available_ta和min_child_idx_ta
    def _cond_0(i, *_):
        return i < time_steps

    def _body_0(i, _min_child_idx_ta):
        # _available_ta = _available_ta.write(i, 1)
        _min_child_idx_ta = _min_child_idx_ta.write(i, tf.cast(i, tf.float32))
        return i + 1, _min_child_idx_ta

    _, min_child_ta = tf.while_loop(_cond_0, _body_0, [0, min_child_ta])

    # 查找可用的节点
    def track_idx(available_, visited_):

        def is_available(_idx):
            not_visited = tf.equal(visited_[_idx], 0)  # 未计算过
            is_seen = tf.equal(available_[_idx], 1)  # 可见的
            _available = tf.logical_and(not_visited, is_seen)
            return _available

        def cond_1(work_, left_idx_, _):
            return tf.logical_and(left_idx_ < total_len, tf.logical_not(work_))  # work表示可用

        def body_1(work_, left_idx_, _):
            def cond_2(work__, right_idx__):
                return tf.logical_and(right_idx__ < total_len, tf.logical_not(work__))

            def body_2(_, right_idx__):
                # right_idx是否可用
                available__ = is_available(right_idx__)
                work__ = tf.cond(available__,
                                 lambda: tf.equal(parent_idx_ta.read(left_idx_), parent_idx_ta.read(right_idx__)),
                                 lambda: False)
                right_idx__ = tf.cond(work__, lambda: right_idx__, lambda: right_idx__ + 1)
                return work__, right_idx__

            # left_idx 是否可用
            _available = is_available(left_idx_)

            work_, right_idx_ = tf.cond(_available,
                                        lambda: tf.while_loop(cond_2, body_2, loop_vars=[False, left_idx_ + 1]),
                                        lambda: (False, -1))  # False, -1如果left_idx不可以，则right_idx没有意义
            left_idx_ = tf.cond(work_, lambda: left_idx_, lambda: left_idx_ + 1)
            return work_, left_idx_, right_idx_

        _work, _left_idx, _right_idx = tf.while_loop(cond_1, body_1, loop_vars=[False, 0, _])  # 从0开始查找
        return _work, _left_idx, _right_idx

    # 循环构造lstm网络
    def cond(time_, *_):
        return time_ < time_steps - 1

    def step(time_, available_, visited_, min_child_ta_, state_ta_):
        _, l_idx, r_idx = track_idx(available_, visited_)

        # 左侧访问过,右侧访问过
        left_visited = tf.one_hot(l_idx, depth=total_len, dtype=tf.float32)
        right_visited = tf.one_hot(r_idx, depth=total_len, dtype=tf.float32)
        visited_ = visited_ + left_visited + right_visited

        p = parent_idx_ta.read(l_idx)
        p = tf.cast(p, tf.int32)
        p_available = tf.one_hot(p, depth=total_len, dtype=tf.float32)
        available_ = available_ + p_available

        # 左右节点的子节点，确定小的在左侧
        c_l = min_child_ta_.read(l_idx)
        c_r = min_child_ta_.read(r_idx)
        c_p = tf.cond(c_l < c_r, lambda: c_l, lambda: c_r)  # 父节点的child设为两个子节点较小的那个
        min_child_ta_ = min_child_ta_.write(p, c_p)

        # 输入与状态
        # 确定左右子树，以保证较小的叶节点出现在左边，否则交换左右子节点
        l_idx1, r_idx1 = tf.cond(c_l < c_r, lambda: (l_idx, r_idx), lambda: (r_idx, l_idx))

        # 如果idx小于句长，则表明是叶节点，输入为单词，状态为0；或如果大于句长，表明为中间节点，输入为0，状态为记录的状态
        l_word, l_state = tf.cond(l_idx1 < time_steps,
                          lambda: (words_ta.read(l_idx1), tf.zeros([batch_size, hidden_size * 2], dtype=tf.float32)),
                          lambda: (tf.zeros([batch_size, embedding_size], dtype=tf.float32), state_ta_.read(l_idx1 - time_steps)))

        r_word, r_state = tf.cond(r_idx1 < time_steps,
                          lambda: (words_ta.read(r_idx1), tf.zeros([batch_size, hidden_size * 2], dtype=tf.float32)),
                          lambda: (tf.zeros([batch_size, embedding_size], dtype=tf.float32), state_ta_.read(r_idx1 - time_steps)))

        # 计算节点状态
        inputs = (tf.cast(l_word, tf.float32), tf.cast(r_word, tf.float32))
        h0, c0 = tf.split(l_state, num_or_size_splits=2, axis=1)
        h1, c1 = tf.split(r_state, num_or_size_splits=2, axis=1)
        states = (h0, c0, h1, c1)
        # 连接节点，主要的计算在这里
        new_h, new_c = narry_lstm_cell.call(inputs, states)
        
        # 写入状态
        state_ta_ = state_ta_.write(p - time_steps, tf.concat([new_h, new_c], 1))

        return time_ + 1, available_, visited_, min_child_ta_, state_ta_

    _, available, visited, min_child_ta, states_ta = \
        tf.while_loop(cond, step, loop_vars=[0, available, visited, min_child_ta, states_ta])

    final_states = states_ta.read(time_steps - 2)
    h, c = tf.split(final_states, 2, 1)

    return h, c


def main():

    p_arr = [15, 15, 14, 13, 10, 10, 9, 9, 12, 11, 11, 12, 13, 14, 16, 16, -1]
    w_arr = [[[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]]

    p_pl = tf.placeholder(tf.float32, shape=[None], name='p_pl')
    w_pl = tf.placeholder(tf.float32, shape=[1, None, 9], name='w_pl')
    with tf.variable_scope('tree_lstm') as scope:
        narry_cell = NarryLSTMCell(hidden_size=10, embedding_size=9)
        narry_cell.build()
        h, c = tree_lstm(narry_cell, p_pl, 1, w_pl)

    with tf.Session() as tfs:
        tf.global_variables_initializer().run()
        h_, c_ = tfs.run([h, c], feed_dict={p_pl: p_arr, w_pl: w_arr})
        print('h_:', h_)
        print('c_:', c_)


if __name__ == '__main__':
    main()
