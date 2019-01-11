
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
    default_word = '-NAN-'
    for aTree, sentence in zip(pptree_encoding_list, sentence_list):
        extended_sentence = []  # word列表，为一个句子
        encoding_word_dict_list.append(extended_sentence)  # 句子的列表
        idx_atree = 0
        for idx_merged in range(len(merged_tree_encoding)):
            if idx_atree == len(aTree):  # atree 到了结尾，将merged tree对应的节点补充上
                for encoding in merged_tree_encoding[idx_merged:]:
                    extended_sentence.append(default_word)
                break
            if compare_encodings(merged_tree_encoding[idx_merged], aTree[idx_atree]) == -1:  # atree节点更大，遍历
                candidate_word = default_word  # 词
                encoding = merged_tree_encoding[idx_merged]  # 相等，补长
                extended_sentence.append(candidate_word)
            else:
                candidate_word = sentence[idx_atree]  # 词
                encoding = merged_tree_encoding[idx_merged]  # 相等，补长
                # 将补长的encoding作为键，词作为值，构建句子字典
                extended_sentence.append(candidate_word)
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


def generate_parent_list(merged_tree_encoding):
    """
    生成pplist，对合成的树结构
    :param merged_tree_encoding: 合成的树的叶节点编码，如['000'，'001'，'01'，'1']
    :return: pplist, 父节点树
    """
    num_words = len(merged_tree_encoding)
    total_len = num_words * 2 - 1
    encoding_list = [merged_tree_encoding[i] if i < num_words else '-1' for i in range(total_len)]
    pp_list = [-1] * total_len

    available = [1 if i < num_words else 0 for i in range(total_len)]  # 1 available, 0 unavailable

    for total in range(num_words, total_len):
        for i in range(total - 1, -1, -1):
            if available[i] == 0:
                continue
            encoding_l = encoding_list[i]
            for j in range(i - 1, -1, -1):
                if available[j] == 0:
                    continue
                encoding_r = encoding_list[j]
                if encoding_l[:-1] == encoding_r[:-1]:
                    # l 与 r为兄弟节点
                    pp_list[i] = total
                    pp_list[j] = total
                    available[i] = 0
                    available[j] = 0

                    encoding_list[total] = encoding_l[:-1]
                    available[total] = 1
                    break
            if available[total] == 1:
                break
    return pp_list


def reform_pp_tree(pp_tree):
    total_len = len(pp_tree)
    word_len = (total_len + 1) / 2
    print('word len', word_len)
    visiable = [1] * int(word_len)
    visited = [0] * int(word_len)
    p_s_dict = dict()

    sequence = []
    count = 0
    while True:
        i = 0
        while i < word_len:
            if visited[i] == 1:
                continue

            p = pp_tree[i]
            if p not in p_s_dict:
                p_s_dict[p] = i
            else:
                sequence.append(p_s_dict[p])
                sequence.append(i)  # 左右节点加入处理序列
                p2 = pp_tree[p]
                p_s_dict[p2] = p  # 更新dict
                # 两个节点访问过
                visited[p_s_dict[i]] = 1
                visited[i] = 1
                count = count + 1
            if count == word_len - 1:
                break
        if count == word_len - 1:
            break

    return sequence


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


def main2():
    encoding_list = ['00', '010', '011', '10', '110', '1110', '11110', '111110', '111111']
    pp_list = generate_parent_list(encoding_list)
    print('pp_list:', pp_list)


def main3():
    pp_list = [15, 15, 14, 13, 10, 10, 9, 9, 12, 11, 11, 12, 13, 14, 16, 16, -1]
    sequence = reform_pp_tree(pp_list)
    print(sequence)


if __name__ == '__main__':
    main3()
