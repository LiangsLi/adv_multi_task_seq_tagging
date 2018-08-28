# -*- coding: utf-8 -*-
import csv
import pathlib

# from .config import MAX_LEN, WORD_VEC_100
# from .config import TRAIN_DATA_MT, TRAIN_PATH, DEV_DATA_MT, DEV_PATH, TEST_DATA_MT, TEST_PATH, WORD_DICT
# from .config import TRAIN_DATA_BI, DEV_DATA_BI, TEST_DATA_BI, TRAIN_DATA_UNI, DEV_DATA_UNI, TEST_DATA_UNI, WORD_SINGLE

from voc import Vocab, Tag


# from voc import Vocab, Tag
# import argparse


class Data_index(object):
    def __init__(self, Vocabs, tags):
        self.VOCABS = Vocabs
        self.TAGS = tags
    
    def to_index_bi(self, words, tags):
        word_idx = []
        words.append('<EOS>')
        words.append('<EOS>')
        words.insert(0, '<BOS>')
        words.insert(0, '<BOS>')
        for i in range(2, len(words) - 2):
            for j in range(-2, 3):
                if words[i + j] in self.VOCABS.word2idx:
                    word_idx.append(self.VOCABS.word2idx[words[i + j]])
                else:
                    word_idx.append(self.VOCABS.word2idx['<OOV>'])
            for j in range(-2, 2):
                if words[i + j] + words[i + j + 1] in self.VOCABS.word2idx:
                    word_idx.append(self.VOCABS.word2idx[words[i + j] + words[i + j + 1]])
                else:
                    word_idx.append(self.VOCABS.word2idx['<OOV>'])
        
        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]
        
        return ','.join(map(str, word_idx)), ','.join(map(str, tag_idx))
    
    def to_index(self, words, tags):
        word_idx = []
        for word in words:
            if word in self.VOCABS.word2idx:
                word_idx.append(self.VOCABS.word2idx[word])
            else:
                word_idx.append(self.VOCABS.word2idx['<OOV>'])
        
        tag_idx = [self.TAGS.tag2idx[tag] for tag in tags]
        
        return ','.join(map(str, word_idx)), ','.join(map(str, tag_idx))
    
    def process_file(self, path, output, bigram=False):
        """
        
        :param path: 原数据的路径
        :param output:  csv writer 对象
        :param bigram: ？？？
        :return:
        """
        # todo:
        src_data, data, label = self.process_data(path)
        for words, tags in zip(data, label):
            length = len(words)
            ratio = (length - 1) / MAX_LEN
            for i in range(0, ratio + 1):
                tmpwords = words[MAX_LEN * i:MAX_LEN * (i + 1)]
                tmptags = tags[MAX_LEN * i:MAX_LEN * (i + 1)]
                if bigram:
                    word_idx, tag_idx = self.to_index_bi(tmpwords, tmptags)
                    length = len(tmpwords) - 4
                else:
                    word_idx, tag_idx = self.to_index(tmpwords, tmptags)
                    length = len(tmpwords)
                output.writerow([word_idx, tag_idx, length])
    
    def process_all_data(self, output_path, input_path, bigram=False, multitask=False):
        def process_file_new(input_file, output):
            with open(input_file, 'r', encoding='utf-8')as f:
                for line in f:
                    temp = line.strip('\n').split('###')
                    word_idxs, tag_idxs = self.to_index(temp[0].split('||'), temp[1].split('||'))
                    length = len(temp[0].split('||'))
                    assert len(temp[1].split('||')) == length
                    output.writerow([word_idxs, tag_idxs, length])
        
        # open file to write
        # if bigram is False:
        #     f_train = open(TRAIN_DATA_UNI, 'w')
        #     f_dev = open(DEV_DATA_UNI, 'w')
        #     f_test = open(TEST_DATA_UNI, 'w')
        # elif multitask:
        #     f_train = open(TRAIN_DATA_MT, 'w')
        #     f_dev = open(DEV_DATA_MT, 'w')
        #     f_test = open(TEST_DATA_MT, 'w')
        # else:
        #     f_train = open(TRAIN_DATA_BI, 'w')
        #     f_dev = open(DEV_DATA_BI, 'w')
        #     f_test = open(TEST_DATA_BI, 'w')
        output_path = pathlib.Path(output_path)
        input_path = pathlib.Path(input_path)
        f_train = open(str(output_path / 'train.csv'), 'w', encoding='utf-8')
        f_dev = open(str(output_path / 'dev.csv'), 'w', encoding='utf-8')
        # f_test = open(TEST_DATA_BI, 'w')
        #  创建 csv writer 对象，三列：words、 tags、 length
        output_train = csv.writer(f_train)
        output_train.writerow(['words', 'tags', 'length'])
        output_dev = csv.writer(f_dev)
        output_dev.writerow(['words', 'tags', 'length'])
        # output_test = csv.writer(f_test)
        # output_test.writerow(['words', 'tags', 'length'])
        # 将数据写入 csv 文件
        process_file_new(str(input_path / 'train.data'), output_train)
        process_file_new(str(input_path / 'test.data'), output_dev)
        # self.process_file(TEST_PATH, output_test, bigram)
    
    def process_data(self, path):
        """
        
        :param path: 原数据路径
        :return: 原数据句子，逗号分割的单字，tags
        """
        src_data = []
        data = []
        label = []
        
        src_data_sentence = []
        data_sentence = []
        label_sentence = []
        
        f = open(path, 'r', encoding='utf-8')
        li = f.readlines()
        f.close()
        
        for line in li:
            # line = unicode(line, 'utf-8')
            # 替换空白字符并切分（基于双空格切分）
            line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
            if len(line_t) < 3:
                # 如果是短于3部分的句子
                if len(data_sentence) == 0:
                    continue
                src_data.append(src_data_sentence)
                data.append(data_sentence)
                label.append(label_sentence)
                src_data_sentence = []
                data_sentence = []
                label_sentence = []
                continue
            # 先是原数据句子
            src_word = line_t[0]
            # 然后是，分割的单字（单词）
            word = line_t[1]
            src_data_sentence.append(src_word)
            data_sentence.append(word)
            # 最后是 tags
            label_sentence += [line_t[2].split('_')[0]]
        
        return src_data, data, label


if __name__ == '__main__':
    VOCABS = Vocab('/Users/liangs/Codes/insurance_data/insurance_wordvec.wv',
                   '../data/20_data/vocab.txt',
                   single_task=False,
                   bi_gram=True,
                   frequency=0)
    TAGS = Tag()  # tag2idx
    init_embedding = VOCABS.word_vectors  # word/char embedding
    da_idx = Data_index(VOCABS, TAGS)  #
    for i in range(1, 21):
        path = "../data/20_data/" + str(i)
        da_idx.process_all_data(path, path, True, multitask=False)
