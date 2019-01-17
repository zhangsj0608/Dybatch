# import os, sys
class Node (object):

    def __init__(self, parent=None, left=None, right=None, is_leaf=False, word=None, label=None):
        self._parent = parent
        self._left = left
        self._right = right
        self._is_leaf = is_leaf
        self._binary_encoding = None
        self._word = word
        self._depth = [1, 1]
        self._label = label

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        self._left = left

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right):
        self._right = right

    @property
    def word(self):
        return self._word

    @word.setter
    def word(self, word):
        self._word = word

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def binary_encoding(self):
        return self._binary_encoding

    @binary_encoding.setter
    def binary_encoding(self, b_en):
        self._binary_encoding = b_en

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, _depth):
        self._depth = _depth

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    # def lr_exchange(self):
    #     tmp = self.left
    #     self.left = self.right
    #     self.right = tmp


class BinaryTree(object):

    def __init__(self, root):
        self.root = root
        self._leaves = list()
        self._leaves_encodings = list()

    @property
    def leaves_nodes(self):
        return self._leaves

    @leaves_nodes.setter
    def leaves_nodes(self, leaves_list):
        self._leaves = leaves_list

    @property
    def leaves_str(self):
        return [node.word for node in self._leaves]

    def traverse(self):
        self.root.binary_encoding = ''
        self._traverse_from_node(self.root)

    def _traverse_from_node(self, node):
        """only keep leaves in the list"""
        # print(node.depth)
        if node.is_leaf:
            self._leaves.append(node)
            self._leaves_encodings.append(node.binary_encoding)
            return

        if node.left is not None:
            node.left.binary_encoding = node.binary_encoding + '0'  # 左子树加0
            self._traverse_from_node(node.left)
        if node.right is not None:
            node.right.binary_encoding = node.binary_encoding + '1'  # 右子树加1
            self._traverse_from_node(node.right)

    def traverse_with_lrchange(self):
        self.root.binary_encoding = ''
        self._traverse_from_node_with_lrchange(self.root)

    def _traverse_from_node_with_lrchange(self, node):
        """
        遍历树，且保留叶子节点，如果右子树为叶节点，左子树为非叶节点，此处将左右子树互换，
        :param node:
        :return:
        """
        if node.is_leaf:
            self._leaves.append(node)
            self._leaves_encodings.append(node.binary_encoding)
            return

        if node.depth[0] > 2 and node.depth[1] == 2:  # 左为非叶子节点，右为叶子节点
            tmp = node.left
            node.left = node.right
            node.right = tmp

        if node.left is not None:
            node.left.binary_encoding = node.binary_encoding + '0'
            self._traverse_from_node_with_lrchange(node.left)
        if node.right is not None:
            node.right.binary_encoding = node.binary_encoding + '1'
            self._traverse_from_node_with_lrchange(node.right)

    @property
    def leaves_encodings(self):
        return self._leaves_encodings


class PPTree(BinaryTree):

    def __init__(self, root=None, pp_list=None, word_list=None, label_list=None):
        """
        initialization of pp tree for Stanford Sentiment Treebank sentence sample
        Args:
            pp_list: list of number representing the node index of the tree
            word_list: the list of words representing the sentence
        """
        super(PPTree, self).__init__(root)
        self.pp_list = pp_list
        self._word_list = word_list
        self._label_list = label_list
        self._node_list = None

    def construct_tree(self):
        assert len(self.pp_list) == 2 * len(self._word_list) - 1, 'the number of tree nodes or number of ' + \
                                                                 'words are not correct'
        pp_list = [int(i)-1 for i in self.pp_list]  # 父节点的序数（1-n）
        node_list = [None] * len(pp_list)  # 节点列表

        words_len = len(word_list)
        for i in range(words_len):

            new_node = Node(word=self._word_list[i], label=self._label_list[i], is_leaf=True)
            node_list[i] = new_node

            c_idx = i  # 当前节点idx
            while True:
                p_idx = pp_list[c_idx]  # 父节点idx
                current_node = node_list[c_idx]
                current_depth = current_node.depth[0] if current_node.depth[0] > current_node.depth[1] else \
                    current_node.depth[1]
                if p_idx == -1:
                    break
                if node_list[p_idx] is not None:  # 父节点已经访问过，说明当前节点为右节点
                    parent_node = node_list[p_idx]
                    parent_node.right = current_node  # 右子节点
                    parent_node.depth[1] = current_depth + 1

                    current_node.parent = parent_node  # 父节点
                    c_idx = p_idx
                else:  # 父节点没访问过
                    parent_node = Node(label=self._label_list[i])
                    node_list[p_idx] = parent_node
                    parent_node.left = current_node  # 左子节点
                    parent_node.depth[0] = current_depth + 1

                    current_node.parent = parent_node
                    break
        self._node_list = node_list
        self.root = node_list[-1]  # root


if __name__ == '__main__':
    pp_str = '16|15|14|14|12|11|10|10|13|11|12|13|17|15|16|17|0'
    word_str = 'The|Wild|Thornberrys|Movie|is|a|jolly|surprise|.'
    label_list = range(17)

    pp_list = pp_str.split('|')
    word_list = word_str.split('|')

    pp_tree = PPTree(pp_list=pp_list, word_list=word_list, label_list=label_list)
    pp_tree.construct_tree()  # 先建树
    pp_tree.traverse_with_lrchange()  # 再遍历

    depth = [n.depth for n in pp_tree._node_list]
    print(pp_tree.leaves_str)
    print(pp_tree.leaves_encodings)
