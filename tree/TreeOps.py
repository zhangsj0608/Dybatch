
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

            if index_merged_tree == len(merged_tree_leaves_encodings):  # 最大树已经到了，则将当前树的剩余部分复制到最大树
                merged_tree_leaves_encodings.extend(tree_leaves_encodings[leave_encoding_index:])
                break

            merged_encoding = merged_tree_leaves_encodings[index_merged_tree]  # 最大树的节点

            if compare_encodings(leave_encoding, merged_encoding) == -1:  # leave_encoding 小
                merged_tree_leaves_encodings = merged_tree_leaves_encodings[0: index_merged_tree] + [leave_encoding] + \
                    merged_tree_leaves_encodings[index_merged_tree:]  # 插入当前的最大树的位置
            else:  # 相等
                merged_tree_leaves_encodings[index_merged_tree] = compare_encodings(leave_encoding, merged_encoding)
                # 将相等的节点换为长度较长的节点

            index_merged_tree += 1  # 最大树的标记右移

    return merged_tree_leaves_encodings


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
    tree_list[1] = ['0', '100', '101', '110', '1110', '11110', '111110', '111111']
    tree_list[0] = ['00', '010', '011', '10', '110', '1110', '11110', '111110', '111111']
    tree_list[2] = ['0', '100', '101', '110', '1110', '11110', '11111']

    print(construct_merged_tree(tree_list))


if __name__ == '__main__':
    main()
