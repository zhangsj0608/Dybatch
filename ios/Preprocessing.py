import numpy as np

dic_path = '../data/dictionary.txt'
sentiment_path = '../data/sentiment_labels.txt'
sentences_path = '../data/SOStr.txt'
ppt_path = '../data/STree.txt'
dataset_sentences_path = '../data/datasetSentences.txt'
dataset_split_path = '../data/datasetSplit.txt'

merged_train_path = '../data/train_set1.txt'
merged_test_path = '../data/test_set.txt'
merged_dev_path = '../data/dev_set.txt'

train = dict()
test = dict()
dev = dict()


def split_dataset():
    file0 = open(dataset_sentences_path)
    file1 = open(dataset_split_path)
    file0.readline()
    file1.readline()

    dataset = [{}, train, test, dev]  # train, test, dev

    for idx_sentence, idx_split in zip(file0, file1):

        idx0, sentence = idx_sentence.split('\t')
        sentence = sentence.strip()
        idx1, split = idx_split.split(',')
        assert idx0 == idx1, 'id of sentence not equal with id of split'

        split = int(split)
        dt_dict = dataset[split]
        dt_dict[sentence] = []

    file0.close()
    file1.close()


def read_pp_tree():

    file0 = open(sentences_path)
    file1 = open(ppt_path)

    for stnc, pp_tree in zip(file0, file1):
        words_list = stnc.split('|')
        pp_tree_list = pp_tree.split('|')

        sentence = (' '.join(words_list)).strip()
        pp_tree = (' '.join(pp_tree_list)).strip()
        # pp_tree_arr = np.array(pp_tree_list, dtype=np.int32)
        if sentence in train:
            train[sentence].append(pp_tree)
        elif sentence in test:
            test[sentence].append(pp_tree)
        elif sentence in dev:
            dev[sentence].append(pp_tree)
        else:
            print('pp_tree sentences:', sentence)

    file0.close()
    file1.close()


def read_label():

    file0 = open(dic_path)
    file1 = open(sentiment_path)

    file1.readline()

    sentiment = [id_sentiment.split('|')[1] for id_sentiment in file1]
    for line in file0:
        sentence, id = line.split('|')
        id = int(id)

        sentence = sentence.strip()
        if sentence in train:
            train[sentence].append(sentiment[id])
        elif sentence in test:
            test[sentence].append(sentiment[id])
        elif sentence in dev:
            dev[sentence].append(sentiment[id])

    file0.close()
    file1.close()


def write_to_file(path, dataset):

    print('write to path ', path)
    i = 0
    with open(path, 'w') as file:
        for key in dataset:
            value = dataset[key]
            if len(value) < 2:
                continue
            file.write('|'.join([key, value[0], value[1]]))
            i = i + 1
            if i <= 1:
                print('|'.join([key, value[0], value[1]]))
    print('writes ', i, 'elements ')


def processing():
    split_dataset()
    read_pp_tree()
    read_label()

    print(len(train))
    print(len(test))
    print(len(dev))

    write_to_file(merged_train_path, train)
    write_to_file(merged_test_path, test)
    write_to_file(merged_dev_path, dev)


if __name__ == '__main__':
    # if need re-process the dataset, namely produce the train_set, test_set, and dev_set, run the processing.
    # processing()
    pass



