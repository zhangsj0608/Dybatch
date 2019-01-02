
def construct_merged_tree(leaves_encodings_of_pptrees_list):
    """
    construct the public & maximum tree out of the pptree list
    :param leaves_encodings_of_pptrees_list: tree leaves encodings list
    :return: the merged tree leaves encodings， 最大树
    """
    if len(leaves_encodings_of_pptrees_list) > 0:
        merged_tree_leaves_encodings = leaves_encodings_of_pptrees_list[0][:]  # 最大树

    for tree_leaves_encodings in leaves_encodings_of_pptrees_list[1:]:
        index_merged_tree = 0  # 最大树的标记
        for leave_encoding_index in range(len(tree_leaves_encodings)):
            leave_encoding = tree_leaves_encodings[leave_encoding_index]  # 当前的树的节点

            while index_merged_tree < len(merged_tree_leaves_encodings) and\
                    compare_encodings(leave_encoding, merged_tree_leaves_encodings[index_merged_tree]) == -2:
                # leave_encoding 大
                index_merged_tree += 1

            if index_merged_tree == len(merged_tree_leaves_encodings):  # 最大树已经到了，则将当最大树的节点前树的剩余部分复制到最大树
                merged_tree_leaves_encodings.extend(tree_leaves_encodings[leave_encoding_index:])
                break

            merged_encoding = merged_tree_leaves_encodings[index_merged_tree]  #

            if compare_encodings(leave_encoding, merged_encoding) == -1:  # leave_encoding 小
                merged_tree_leaves_encodings = merged_tree_leaves_encodings[0: index_merged_tree] + [leave_encoding] + \
                    merged_tree_leaves_encodings[index_merged_tree:]  # 插入当前的最大树的位置
            else:  # 相等
                common_encoding = compare_encodings(leave_encoding, merged_encoding)
                # 将相等的节点换为长度较长的节点
                merged_tree_leaves_encodings[index_merged_tree] = common_encoding

            index_merged_tree += 1  # 最大树的标记右移

    return merged_tree_leaves_encodings


def extend_tree_encodings(pptree_encoding_list, sentence_list, merged_tree_encoding):
    """
    produce a list of extended tree_encodings for the pptrees with respect to the merged encoding
    :param pptree_encoding_list: a list of elements, each of which is the original leave encodings for a tree
    :param merged_tree_encoding: the merged tree encodings
    :return: a list of elements, each element is a dict representing a tree/sentence.
            encoding is the key, and word is the value in each dict.
    """
    encoding_word_dict_list = list()
    for aTree, sentence in zip(pptree_encoding_list, sentence_list):
        encoding_word_dict = dict()  # encoding-word字典，为一个句子
        encoding_word_dict_list.append(encoding_word_dict)  # 句子的列表
        idx_atree = 0
        for idx_merged in range(len(merged_tree_encoding)):
            if idx_atree == len(aTree):  # atree 到了结尾，将merged tree对应的节点补充上
                for encoding in merged_tree_encoding[idx_merged:]:
                    encoding_word_dict[encoding] = None
                break
            if compare_encodings(merged_tree_encoding[idx_merged], aTree[idx_atree]) == -1:  # atree节点更大，遍历
                candidate_word = None  # 词
                encoding = merged_tree_encoding[idx_merged]  # 相等，补长
                encoding_word_dict[encoding] = candidate_word
            else:
                candidate_word = sentence[idx_atree]  # 词
                encoding = merged_tree_encoding[idx_merged]  # 相等，补长
                # aTree[idx_atree] = encoding
                encoding_word_dict[encoding] = candidate_word  # 将补长的encoding作为键，词作为值，构建句子字典
                idx_atree += 1
    return encoding_word_dict_list


def compare_encodings(str0, str1):
    """
    compare two encoding strings
    :param str0: encoding string made of 0 and 1
    :param str1: encoding string made of 0 and 1
    :return: str0 < str1, 则返回-1；str0 > str1, 则返回-2；如果两个长度不同，且内容相同，则返回长str
    """
    if len(str0) < len(str1):
        str0 += '0' * (len(str1) - len(str0))
    elif len(str0) > len(str1):
        str1 += '0' * (len(str0) - len(str1))

    if str0 == str1:
        return str0  # 扩展后两个相同，返回长的str
    for i in range(len(str0)):
        if str0[i] < str1[i]:  # str0 小
            return -1
        elif str0[i] > str1[i]:  # str0 大
            return -2


def main():
    tree_list = [['']] * 3
    tree_list[0] = ['0', '100', '101', '110', '1110', '11110', '111110', '111111']
    tree_list[1] = ['00', '010', '011', '10', '110', '1110', '11110', '111110', '111111']
    tree_list[2] = ['0', '100', '101', '110', '1110', '11110', '11111']
    
    sentence_list = [['']] * 3
    sentence_list[0] = ['a0', 'b0', 'c0', 'd0', 'e0', 'f0', 'g0', 'h0']
    sentence_list[1] = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1']
    sentence_list[2] = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2']

    merged_tree = construct_merged_tree(tree_list)
    extended_tree_list = extend_tree_encodings(tree_list, sentence_list, merged_tree)

    for i in extended_tree_list:
        print(i)


if __name__ == '__main__':
    main()
