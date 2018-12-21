# import os, sys
class Node (object):

    def __init__(self, parent=None, left=None, right=None, is_leaf=False, word=None):
        self._parent = parent
        self._left = left
        self._right = right
        self._is_leaf = is_leaf
        self._binary_encoding = None
        self._word = word
        self._depth = (1, 1)

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

    def __init__(self, root=None, pp_list=None, word_list=None):
        """
        initialization of pp tree for Stanford Sentiment Treebank sentence sample
        Args:
            pp_list: list of number representing the node index of the tree
            word_list: the list of words representing the sentence
        """
        super(PPTree, self).__init__(root)
        self.pp_list = pp_list
        self.word_list = word_list
        self.tree_node_list = [None] * len(pp_list)  # 树的所有节点，按照pp_list的顺序存放

    def construct_tree(self):
        assert len(self.pp_list) == 2 * len(self.word_list) - 1, 'the number of tree nodes or number of ' + \
                                                                 'words are not correct'
        parent_list = [int(i)-1 for i in self.pp_list]  # 父节点的序数（1-n）
        node_list = self.tree_node_list  # 树节点列表
        # print(parent_list)

        tmp = [(parent_list[i], i) for i in range(len(self.word_list))]  # 长度为句子长，（父节点，子节点）对
        while len(tmp) > 1:
            parent_child = [(None, None)] * len(parent_list)  # 长度为所有序列长，（tmp index, child序数）放在每个parent的index
            for i in range(len(tmp)):
                parent_index = tmp[i][0]
                child_index = tmp[i][1]
                if parent_child[parent_index][1] is None:
                    parent_child[parent_index] = (i, child_index)  # 左边子树出现在tmp的序数，及左子树节点
                else:
                    parent_child_index_right = i  # 右边子树出现在tmp的序数
                    break

            left_index = parent_child[parent_index][1]  # 左边子树节点
            right_index = child_index  # 右子树节点
            # print('Parent index: {0:} left index: {1:} right index: {2:}'.format(parent_index, left_index,
            # right_index))

            parent_child_index_left = parent_child[parent_index][0]  # 左边子树出现在tmp的序数

            if node_list[parent_index] is None:  # node列表中的父节点
                parent_node = Node()
                node_list[parent_index] = parent_node

            if node_list[left_index] is None:  # node列表的左子树
                left_node = Node(parent=parent_node, is_leaf=True, word=self.word_list[left_index])
                node_list[left_index] = left_node
            else:
                left_node = node_list[left_index]

            if node_list[right_index] is None:  # node列表的右子树
                right_node = Node(parent=parent_node, is_leaf=True, word=self.word_list[right_index])
                node_list[right_index] = right_node
            else:
                right_node = node_list[right_index]

            # assign left and right children for parent node
            parent_node.left = left_node
            parent_node.right = right_node

            # assign the left and right depth for parent node
            left_depth = left_node.depth[0] if left_node.depth[0] > left_node.depth[1] else left_node.depth[1]
            right_depth = right_node.depth[0] if right_node.depth[0] > right_node.depth[1] else right_node.depth[1]
            parent_node.depth = (left_depth + 1, right_depth + 1)

            left_node.parent = parent_node
            right_node.parent = parent_node

            tmp[parent_child_index_left] = (parent_list[parent_index], parent_index)  # 加入合成节点后
            tmp.pop(parent_child_index_right)

        self.root = self.tree_node_list[-1]
        return self.root


if __name__ == '__main__':
    pp_str = '16|15|14|14|12|11|10|10|13|11|12|13|17|15|16|17|0'
    word_str = 'The|Wild|Thornberrys|Movie|is|a|jolly|surprise|.'

    pp_list = pp_str.split('|')
    word_list = word_str.split('|')

    pp_tree = PPTree(pp_list=pp_list, word_list=word_list)
    pp_tree.construct_tree()  # 先建树
    pp_tree.traverse_with_lrchange()  # 再遍历

    print(pp_tree.leaves_str)
    print(pp_tree.leaves_encodings)
