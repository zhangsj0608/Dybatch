import numpy as np
import tensorflow as tf
import os
import sys

# from tf_data_utils import extract_tree_data,extract_batch_tree_data


class tf_NarytreeLSTM(object):

    def __init__(self, config):
        self.emb_dim = config.emb_dim  # embedding size
        self.hidden_dim = config.hidden_dim  # hidden size
        self.num_emb = config.num_emb  # 词典的词数
        self.output_dim = config.output_dim  # output size, 类别
        self.config = config
        self.batch_size = config.batch_size  # batch size
        self.reg = self.config.reg
        self.degree = config.degree  # 树的维度，即分叉的数量，一般为2
        self.inputs = None
        self.tree_str = None
        self.labels = None
        assert self.emb_dim > 1 and self.hidden_dim > 1

        # 在此处构建图
        self.add_placeholders()  # 构建输入的placeholder
        emb_leaves = self.add_embedding()  # 增加embedding层
        self.add_model_variables()  # 构建模型的变量
        batch_loss = self.compute_loss(emb_leaves)  # 计算模型
        self.loss, self.total_loss = self.calc_batch_loss(batch_loss)  # 将batch的loss累加到一起
        self.train_op1, self.train_op2 = self.add_training_op()  # 构建训练op

    def add_embedding(self):
        # TODO 读入词典，并构造embedding layer

        with tf.variable_scope("Embed",regularizer=None):
            embedding=tf.get_variable('embedding',[self.num_emb,
                                                   self.emb_dim]
                        ,initializer=tf.random_uniform_initializer(-0.05,0.05),trainable=True,regularizer=None)
            ix= tf.to_int32(tf.not_equal(self.inputs, -1)) * self.inputs
            emb_tree=tf.nn.embedding_lookup(embedding,ix)
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.inputs, -1)), 2))

            return emb_tree

    def add_placeholders(self):
        # TODO 构建dataset的输入

        total_length = self.config.maxnodesize
        batch_size = self.config.batch_size  # batch_size

        self.inputs = tf.placeholder(tf.int32, [batch_size, total_length], name='input')
        self.tree_str = tf.placeholder(tf.int32, [batch_size, total_length, 2], name='tree')
        self.labels = tf.placeholder(tf.int32, [batch_size, total_length], name='labels')
        # self.dropout = tf.placeholder(tf.float32,name='dropout')

        # self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.tree_str, -1)), [1, 2])  # 内部节点
        # self.n_inodes = self.n_inodes / 2
        #
        # self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.inputs, -1)), [1])  # 叶节点
        # self.batch_len = tf.placeholder(tf.int32,name="batch_len")

    def calc_wt_init(self, fan_in=300):
        eps = 1.0/np.sqrt(fan_in)
        return eps

    def add_model_variables(self):
        # 增加所有的变量
        with tf.variable_scope("Composition", initializer=tf.contrib.layers.xavier_initializer(), regularizer=
                               tf.contrib.layers.l2_regularizer(self.config.reg)):
            # 用来处理输入
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(), self.calc_wt_init()))
            # 用来处理状态,各个列的部分分别为u,o,i,f1,f2,...,fn，每个部分的维度应该为[n * hidden_dim, hidden_dim]
            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),
                                                                           self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb", [4 * self.hidden_dim], initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))
        with tf.variable_scope("Projection", regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),
                                                                          self.calc_wt_init(self.hidden_dim)))
            bu = tf.get_variable("bu", [self.output_dim],initializer=
                                 tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self, whole_emb, indices, dense_shape):
        """
        句子的词向量作为输入，处理单个句子的叶节点
        :param whole_emb: 词向量tensor，一个batch的所有句子的所有叶节点向量的列表，形状为[sum(len(sentence)), emb_dim]
        :param indices: 代表单词位置的tensor，每个元素表示了单词在状态中的位置， 形状为[sum(len(sentence)), 2]
        :param dense_shape: 构造的状态矩阵的形状[batch_size, len_stn]
        :return: 所有叶节点的输出状态，含c,h,形状为[batch_size, len_stn, 2 * hidden_dim]，和状态的mask,形状为[batch_size,
                n_inode, hidden_dim]
        """
        # 对输入节点，也就是叶节点进行输入和输出计算，emb是所有叶节点词向量的列表
        with tf.variable_scope("Composition", reuse=True):
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            b = tf.slice(cb, [0], [2 * self.hidden_dim])

            concat_uo = tf.matmul(whole_emb, cU) + b  # [sum(len(sentence), 2 * hidden_dim)]
            u, o = tf.split(1, 2, concat_uo)  # 注意U的顺序，为u，o
            o = tf.nn.sigmoid(o)
            u = tf.nn.tanh(u)

            c = u  # tf.squeeze(u)
            h = o * tf.nn.tanh(c)

            hc = tf.concat(1, [h, c])
            hc = tf.squeeze(hc)  # [num_whole_words, 2 * hidden_dim]

        # 将hc整理为状态矩阵
        # 构造sparse_tensor,indices的形状为[batch_size, len_stn]
        num_whole_words = tf.shape(indices)[0]
        sparse_value = tf.range(1.0, num_whole_words + 1)  # 1:n
        dn_indices = tf.sparse_to_dense(sparse_indices=indices, sparse_values=sparse_value, output_shape=dense_shape)
        # 构造states列表
        zero_values = tf.expand_dims(tf.zeros_like(hc[0]), 0)
        emb_values = tf.concat([zero_values, hc], 0)  # 长度为1+num_whole_words
        states = tf.nn.embedding_lookup(emb_values, dn_indices)  # [batch_size, len_stn, 2 * hidden_dim]
        # 构造mask列表
        one_values = tf.ones([1, self.hidden_dim])
        zero_values = tf.onse([1, self.hidden_dim])
        tiled_ones = tf.reshape(tf.tile(one_values, num_whole_words), [num_whole_words, self.hidden_dim])
        emb_values = tf.concat([zero_values, tiled_ones], 0)
        masks = tf.nn.embedding_lookup(emb_values, dn_indices)  # [batch_size, len_stn, hidden_dim]
        # slice长度[batch_size, num_inodes, hidden_dim]的mask
        num_leaves = (tf.shape(dense_shape)[1] + 1) / 2
        num_inodes = tf.shape(dense_shape)[1] - num_leaves
        masks = tf.slice(masks, [0, num_leaves, 0], [self.batch_size, num_inodes, self.hidden_dim])
        return states, masks

    def compute_states(self, states, treestr, masks):
        """
        计算单个句子的状态列表
        :param states: 一个batch的初始状态，含h,c,维度为[batch_size, len_stn, 2 * hidden_dim]
        :param treestr: 一个树的结构，长度为树的内部节点，注意顺序，内容为（l_idx, r_idx）
        :param masks: 一个batch的mask, 只对iNode做mask，维度为[batch_size, n_inodes, hidden_dim]
        :return: 输出的张量，只含h，维度为[batch_size, len_stn, hidden_dim]
        """
        # num_leaves = tf.squeeze(tf.gather(self.num_leaves, idx_batch))  # 一个句子的长度，句子的id为idx_batch
        # n_inodes = tf.gather(self.n_inodes, idx_batch)  # 一个句子中非叶节点的数量
        # # 一个句子中的词向量，注意emb对前n个词进行了提取，说明输入是在后边补了0，维度[num_leaves, emb_dim]
        # embx = tf.gather(tf.gather(emb, idx_batch), tf.range(num_leaves))
        # treestr = tf.gather(tf.gather(self.tree_str, idx_batch), tf.range(n_inodes))  # 句子的内节点，维度[n_inodes, 2]
        # leaf_hc = self.process_leafs(embx)  # 叶节点的输出，维度为[num_leaves, 2 * hidden_dim]
        states = tf.transpose(states, perm=[1, 0, 2])  # len_stn, batch_size, 2 * hidden_dim
        masks = tf.transpose(masks, perm=[1, 0, 2])  # n_inodes, batch_size, hidden_dim
        initial_h, initial_c = tf.split(states, num_or_size_splits=2, axis=2)

        n_inodes = tf.shape(treestr)[0]  # 除掉叶子节点的个数
        num_leaves = n_inodes + 1  # 叶子节点的个数
        # 最后的所有节点的输出，现在只有叶节点部分 [num_leaves, batch_size, hidden_dim]
        node_h = tf.slice(initial_h, [0, 0, 0], [self.batch_size, num_leaves, self.hidden_dim])
        node_c = tf.slice(initial_c, [0, 0, 0], [self.batch_size, num_leaves, self.hidden_dim])

        idx_var = tf.constant(0)

        with tf.variable_scope("Composition", reuse=True):

            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            bu, bo, bi, bf = tf.split(0, 4, cb)

            def _recurrence(node_h, node_c, idx_var):
                node_info = tf.gather(treestr, idx_var)  # 可能是左右节点的编号
                mask = tf.gather(masks, idx_var)  # mask

                left_h = tf.gather(node_h, node_info[0])  # 左右节点的h,维度[batch_size, hidden_dim]
                right_h = tf.gather(node_h, node_info[1])
                child_h = tf.concat([left_h, right_h], axis=1)  # 维度[batch_size, hidden_dim * 2]

                left_c = tf.gather(node_c, node_info[0])  # 左右节点的c,维度[batch_size, hidden_dim]
                right_c = tf.gather(node_c, node_info[1])

                tmp = tf.matmul(child_h, cW)  # 维度为[batch_size, hidden_dim * 5]
                u, o, i, fl, fr = tf.split(tmp, 5, 1)  # 注意w的顺序，即u,o,i,f1,f2

                i = tf.nn.sigmoid(i + bi)  # 维度均为[batch_size, hidden_dim]
                o = tf.nn.sigmoid(o + bo)
                u = tf.nn.tanh(u + bu)
                fl = tf.nn.sigmoid(fl + bf)
                fr = tf.nn.sigmoid(fr + bf)

                c = i * u + fl * left_c + fr * right_c  # [batch_size, hidden_dim]
                h = o * tf.nn.tanh(c)  # [batch_size, hidden_dim]

                ini_h = tf.gather(initial_h, idx_var)  # 初始化的h
                ini_c = tf.gather(initial_c, idx_var)  # 初始化的c

                new_h = h * (1 - mask) + ini_h * mask  # 去掉每次前部分计算的部分，而加上初始值
                new_c = c * (1 - mask) + ini_c * mask

                node_h = tf.concat([node_h, new_h], 0)  # 这里每输出一个状态，则在整个的状态上的末尾加上该状态
                node_c = tf.concat([node_c, new_c], 0)

                idx_var = tf.add(idx_var, 1)
                return node_h, node_c, idx_var

            loop_cond = lambda a1, b1, idx_var: tf.less(idx_var, n_inodes)

            loop_vars = [node_h, node_c, idx_var]
            node_h, node_c, idx_var = tf.while_loop(loop_cond, _recurrence,
                                                    loop_vars, parallel_iterations=10)
            return node_h

    def create_output(self, tree_states):
        """
        计算映射层的输出，对整个句子进行处理
        :param tree_states: 句子的状态列表的张量，维度为[num_leaves + n_inodes, hidden_dim]
        :return: 所有节点的输出，维度[num_leaves + n_inodes, output_dim]
        """
        with tf.variable_scope("Projection", reuse=True):

            U = tf.get_variable("U", [self.output_dim, self.hidden_dim])
            bu = tf.get_variable("bu", [self.output_dim])

            h = tf.matmul(tree_states, U, transpose_b=True)+bu
            return h

    def calc_loss(self, logits, labels):
        """
        计算一个句子的loss,即每个节点的loss之和
        :param logits: 预测输出，维度为[num_leaves + i_nodes, num_outputs]
        :param labels: 真正标签，维度为[num_leaves + i_nodes, num_outputs],每行为one-hot向量
        :return:
        """
        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        loss = tf.reduce_sum(l1, [0])
        return loss

    def compute_loss(self, emb_batch, curr_batch_size=None):
        """
        计算一个batch的loss
        :param emb_batch: 一个batch的句子对应的词向量，维度为[batch_size, length, emb_size]
        :param curr_batch_size:
        :return: batch loss，维度为[batch_size, 1]
        """
        outloss=[]
        prediction=[]
        for idx_batch in range(self.config.batch_size):
            # 处理单个句子
            tree_states = self.compute_states(emb_batch, idx_batch)
            logits = self.create_output(tree_states)  # 句子所有节点的输出，维度[num_leaves + i_nodes, output_dim]
            # 句子所有的label，即每个树的节点一个one-hot的label，形状为[num_leaves + i_nodes, output_dim]
            labels1 = tf.gather(self.labels, idx_batch)  # labels1是比较长的，有空字符用-1表示
            labels2 = tf.reduce_sum(tf.to_int32(tf.not_equal(labels1, -1)))  # 提取一个句子中真正的label数量
            labels = tf.gather(labels1, tf.range(labels2))  # 真正的label张量，形状同上
            loss = self.calc_loss(logits, labels)

            pred = tf.nn.softmax(logits)  # 预测值，经过了softmax
            pred_root = tf.gather(pred, labels2-1)  # 根节点的预测值，也即整个句子的预测,维度为[output_dim]

            prediction.append(pred_root)
            outloss.append(loss)

        batch_loss = tf.pack(outloss)  # 维度[batch_size, 1]
        self.pred = tf.pack(prediction)  # 维度[batch_size, output_dim]

        return batch_loss

    def calc_batch_loss(self, batch_loss):
        """
        计算batch_loss，即一个batch中的loss的平均值
        :param batch_loss: 一个batch的loss，维度为[batch_size, 1]
        :return: batch_loss, 维度为[0], total_loss 加上了正规化的loss，维度为[0]
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        loss = tf.reduce_mean(batch_loss)
        total_loss = loss+0.5*regpart
        return loss, total_loss

    def add_training_op_old(self):
        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op

    def add_training_op(self):
        # TODO 只保留模型参数训练的op
        """
        构建训练的操作
        :return: 训练模型参数和训练词向量的参数的操作列表
        """
        loss = self.total_loss
        opt1 = tf.train.AdagradOptimizer(self.config.lr)
        opt2 = tf.train.AdagradOptimizer(self.config.emb_lr)

        # 对词向量和模型参数的梯度
        ts = tf.trainable_variables()
        gs = tf.gradients(loss, ts)
        gs_ts = zip(gs, ts)

        gt_emb, gt_nn = [], []
        for g, t in gs_ts:  # 梯度和变量
            #print t.name,g.name
            if "Embed/embedding:0" in t.name:  # 变量为词向量
                gt_emb.append((g, t))
            else:  # 变量为模型参数
                gt_nn.append((g, t))

        train_op1 = opt1.apply_gradients(gt_nn)
        train_op2 = opt2.apply_gradients(gt_emb)  # 更新参数
        train_op = [train_op1, train_op2]

        return train_op

    def train(self, data, sess):
        """
        训练模型
        :param data: 所有数据
        :param sess: tf.Session
        :return: 所有batch的loss的平均值
        """
        from random import shuffle
        data_idxs = range(len(data))
        shuffle(data_idxs)
        losses = []
        for i in range(0, len(data), self.batch_size):
            # 提取一个batch的数据
            batch_size = min(i + self.batch_size, len(data)) - i
            if batch_size < self.batch_size:
                break

            batch_idxs = data_idxs[i: i+batch_size]
            batch_data = [data[ix] for ix in batch_idxs]

            # 提取数据中的输入， 树结构，和标签
            input_b, treestr_b, labels_b = extract_batch_tree_data(batch_data, self.config.maxnodesize)
            feed = {self.inputs: input_b, self.tree_str: treestr_b, self.labels: labels_b,
                    self.dropout: self.config.dropout, self.batch_len: len(input_b)}

            loss, _, _ = sess.run([self.loss, self.train_op1, self.train_op2], feed_dict=feed)
            losses.append(loss)

            avg_loss = np.mean(losses)  # 当前的平均batch_loss
            sstr = 'avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            sys.stdout.write(sstr)
            sys.stdout.flush()
        return np.mean(losses)

    def evaluate(self, data, sess):
        """
        预测函数
        :param data: 所有的数据
        :param sess:  tf.session
        :return:
        """

        num_correct = 0
        total_data = 0
        data_idxs = range(len(data))
        test_batch_size = self.config.batch_size
        losses=[]
        for i in range(0, len(data), test_batch_size):
            batch_size = min(i+test_batch_size, len(data))-i
            if batch_size < test_batch_size:
                break
            batch_idxs = data_idxs[i:i+batch_size]
            batch_data = [data[ix] for ix in batch_idxs]
            labels_root = [l for _, l in batch_data]
            input_b, treestr_b, labels_b = extract_batch_tree_data(batch_data, self.config.maxnodesize)

            feed = {self.inputs: input_b, self.tree_str: treestr_b, self.labels: labels_b,
                    self.dropout: 1.0, self.batch_len: len(input_b)}

            pred_y = sess.run(self.pred, feed_dict=feed)
            #print pred_y,labels_root
            y = np.argmax(pred_y, axis=1)
            #num_correct+=np.sum(y==np.array(labels_root))
            for i, v in enumerate(labels_root):
                if y[i] == v:
                    num_correct = num_correct + 1
                total_data = total_data + 1
            #break

        acc = float(num_correct) / float(total_data)
        return acc


class tf_ChildsumtreeLSTM(tf_NarytreeLSTM):
    def add_model_variables(self):
        with tf.variable_scope("Composition",
                                initializer=
                                tf.contrib.layers.xavier_initializer(),
                                regularizer=
                                tf.contrib.layers.l2_regularizer(self.config.reg
            )):

            cUW = tf.get_variable("cUW",[self.emb_dim+self.hidden_dim,4*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

        with tf.variable_scope("Projection",regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                initializer=tf.random_uniform_initializer(
                                    -0.05,0.05))
            bu = tf.get_variable("bu",[self.output_dim],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self,emb):

        with tf.variable_scope("Composition",reuse=True):
            cUW = tf.get_variable("cUW")
            cb = tf.get_variable("cb")
            U = tf.slice(cUW,[0,0],[self.emb_dim,2*self.hidden_dim])
            b = tf.slice(cb,[0],[2*self.hidden_dim])
            def _recurseleaf(x):

                concat_uo = tf.matmul(tf.expand_dims(x,0),U) + b
                u,o = tf.split(1,2,concat_uo)
                o=tf.nn.sigmoid(o)
                u=tf.nn.tanh(u)

                c = u#tf.squeeze(u)
                h = o * tf.nn.tanh(c)


                hc = tf.concat(1,[h,c])
                hc=tf.squeeze(hc)
                return hc

            hc = tf.map_fn(_recurseleaf,emb)
            return hc


    def compute_states(self,emb,idx_batch=0):

        #if num_leaves is None:
            #num_leaves = self.n_nodes - self.n_inodes
        num_leaves = tf.squeeze(tf.gather(self.num_leaves,idx_batch))
        #num_leaves=tf.Print(num_leaves,[num_leaves])
        n_inodes = tf.gather(self.n_inodes,idx_batch)
        #embx=tf.gather(emb,tf.range(num_leaves))
        emb_tree=tf.gather(emb,idx_batch)
        emb_leaf=tf.gather(emb_tree,tf.range(num_leaves))
        #treestr=self.treestr#tf.gather(self.treestr,tf.range(self.n_inodes))
        treestr=tf.gather(tf.gather(self.tree_str, idx_batch), tf.range(n_inodes))
        leaf_hc = self.process_leafs(emb_leaf)
        leaf_h,leaf_c=tf.split(1,2,leaf_hc)

        node_h=tf.identity(leaf_h)
        node_c=tf.identity(leaf_c)

        idx_var=tf.constant(0) #tf.Variable(0,trainable=False)

        with tf.variable_scope("Composition",reuse=True):

            cUW = tf.get_variable("cUW",[self.emb_dim+self.hidden_dim,4*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])
            bu,bo,bi,bf=tf.split(0,4,cb)

            UW = tf.slice(cUW,[0,0],[-1,3*self.hidden_dim])

            U_fW_f=tf.slice(cUW,[0,3*self.hidden_dim],[-1,-1])

            def _recurrence(emb_tree,node_h,node_c,idx_var):
                node_x=tf.gather(emb_tree,num_leaves+idx_var)
                #node_x=tf.zeros([self.emb_dim])
                node_info=tf.gather(treestr,idx_var)

                child_h=tf.gather(node_h,node_info)
                child_c=tf.gather(node_c,node_info)

                concat_xh=tf.concat(0,[node_x,tf.reduce_sum(node_h,[0])])

                tmp=tf.matmul(tf.expand_dims(concat_xh,0),UW)
                u,o,i=tf.split(1,3,tmp)
                #node_x=tf.Print(node_x,[tf.shape(node_x),node_x.get_shape()])
                hl,hr=tf.split(0,2,child_h)
                x_hl=tf.concat(0,[node_x,tf.squeeze(hl)])
                x_hr=tf.concat(0,[node_x,tf.squeeze(hr)])
                fl=tf.matmul(tf.expand_dims(x_hl,0),U_fW_f)
                fr=tf.matmul(tf.expand_dims(x_hr,0),U_fW_f)

                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.tanh(u+bu)
                fl=tf.nn.sigmoid(fl+bf)
                fr=tf.nn.sigmoid(fr+bf)

                f=tf.concat(0,[fl,fr])
                c = i * u + tf.reduce_sum(f*child_c,[0])
                h = o * tf.nn.tanh(c)

                node_h = tf.concat(0,[node_h,h])

                node_c = tf.concat(0,[node_c,c])

                idx_var=tf.add(idx_var,1)

                return emb_tree,node_h,node_c,idx_var
            loop_cond = lambda a1,b1,c1,idx_var: tf.less(idx_var,n_inodes)

            loop_vars=[emb_tree,node_h,node_c,idx_var]
            emb_tree,node_h,node_c,idx_var=tf.while_loop(loop_cond, _recurrence, loop_vars, parallel_iterations=1)

            return node_h