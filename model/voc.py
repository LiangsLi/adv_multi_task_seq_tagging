import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec


class Vocab(object):
    """生成最后的 embedding 向量，包括单词和单字，保存在 word vectors 中，numpy 数组形式;
        word2idx 保存在word2idx 中； 所有单词保存在 table 中"""
    
    def __init__(self, path_vec, train_word, single_task, bi_gram, frequency=0):
        """
        返回预训练词向量
        :param path_vec: 预训练词向量文件
        :param train_word: 单词表文件
        :param single_task: 是否多任务
        :param bi_gram: 是否使用 bigram 特征
        :param frequency: 最小词频
        """
        self.path = path_vec
        self.table_path = train_word
        self.word2idx = defaultdict(int)
        self.word_vectors = None
        self.single = single_task
        self.bigram = bi_gram
        self.frequency = frequency
        self.table = []
        self.process_table(self.table_path, self.single)
        # 单词表中的每一行都保存在table中，每一行都只有一个单词
        self.load_data(file_type='gensim')
    
    def process_table(self, word_path, single):
        """
        
        :param word_path: 单词表文件
        :param single: 是否多任务
        :return:
        """
        if self.bigram:
            # 如果使用 bi-gram：
            f = open(word_path, 'r', encoding='utf-8')
            text = f.readlines()
            if single is False:
                # 如果不是单任务：
                for line in text:
                    com = line.strip()
                    # 将单词逐个添加到 table
                    self.table.append(com)
            else:
                # 如果是单任务
                # pass
                raise RuntimeError("single must be False")
                # table = []
                # for line in text:
                #     com = line.strip().split(' ')
                #     if len(com[1]) == 2 and int(com[2]) > self.frequency:
                #         table.append(com[1])
                # self.table = set(table)
            f.close()
        else:
            self.table = set()
    
    def load_data(self, file_type='txt'):
        if file_type == 'txt':
            with open(self.path, 'r', encoding='utf-8') as f:
                # 处理词向量 txt 的首行，（数量，维度）
                line = f.readline().strip().split(" ")
                N, dim = map(int, line)
                # 读取词向量文件，加载为一个 numpy 数组
                # 注意这里的词向量文件实际是字向量，只有单字的向量
                self.word_vectors = []
                idx = 0
                for k in range(N):
                    line = f.readline().strip().split(" ")
                    # line[0]是单词
                    self.word2idx[line[0]] = idx
                    # 生成 Numpy 数组
                    vector = np.asarray(map(float, line[1:]), dtype=np.float32)
                    self.word_vectors.append(vector)
                    idx += 1
                # 统计单词表中的单词的词向量（裁剪词向量）
                # 由于我们只有单字的向量形式，所以需要拼凑出词语的向量形式
                # 这里使用的方式非常简单，简单累加之后除以2.0
                count = 0
                for word in self.table:
                    mean_vec = np.zeros(dim)
                    for ch in word:
                        # 逐个单字处理
                        if ch in self.word2idx:
                            mean_vec += self.word_vectors[self.word2idx[ch]]
                            count += 1
                        else:
                            mean_vec += self.word_vectors[self.word2idx['<OOV>']]
                    # 由字向量简单组合得到词向量
                    word_vec = mean_vec / 2.0
                    self.word2idx[word] = idx
                    self.word_vectors.append(word_vec)
                    idx += 1
                
                # print 'Vocab size:', len(self.word_vectors)
                # print 'word2idx:', len(self.word2idx)
                # print 'index:', idx
                # print 'count:', count
                # 这样最后 word vectors 中，既有单词的向量，也有所有单字的向量
        elif file_type == 'gensim':
            self.word_vectors = []
            model = Word2Vec.load(self.path)
            pre_train_words = model.wv.index2word
            for idx, word in enumerate(self.table):
                if word in pre_train_words:
                    vector = model.wv[word]
                    self.word_vectors.append(vector)
                else:
                    randn_vec = np.random.randn(100)
                    self.word_vectors.append(randn_vec)
                self.word2idx[word] = idx
        
        else:
            raise RuntimeError("bad pre-trained embedding file type")
        self.word_vectors = np.asarray(self.word_vectors, dtype=np.float32)


class Tag(object):
    def __init__(self):
        # self.tag2idx = defaultdict(int)
        self.tag2idx = dict()
        self.define_tags()
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
    
    def define_tags(self):
        self.tag2idx['O'] = 0
        self.tag2idx['B'] = 1
        self.tag2idx['M'] = 2
        self.tag2idx['E'] = 3
        self.tag2idx['I'] = 4
