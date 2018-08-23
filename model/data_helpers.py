import pandas as pd
import numpy as np


class BucketedDataIterator(object):
    def __init__(self, df, num_buckets=10):
        self.df = df
        self.total = len(df)
        # 按照句子长度排序：
        df_sort = df.sort_values('length').reset_index(drop=True)
        self.size = self.total / num_buckets
        self.dfs = []
        for bucket in range(num_buckets - 1):
            self.dfs.append(df_sort.ix[bucket * self.size: (bucket + 1) * self.size - 1])
        self.dfs.append(df_sort.ix[(num_buckets - 1) * self.size:])
        self.num_buckets = num_buckets
        
        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.pos = 0
        self.shuffle()
        
        self.epochs = 0
    
    def shuffle(self):
        # sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
    
    def next_batch(self, batch_size, bigram=False, round=-1, classifier=False):
        """
        
        :param batch_size: batch 大小
        :param bigram: ???
        :param round: 指明什么分类（用于对抗训练）
        :param classifier: 是否返回分类（用于对抗训练）
        :return:
        """
        if np.any(self.cursor + batch_size + 1 > self.size):
            self.epochs += 1
            self.shuffle()
        
        i = np.random.randint(0, self.num_buckets)
        
        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + batch_size - 1]
        # words = map(lambda x: map(int, x.split("||")), res['words'].tolist())
        # tags = map(lambda x: map(int, x.split("||")), res['tags'].tolist())
        words = [[int(x) for x in y.split(',')] for y in res['words'].tolist()]
        tags = [[int(x) for x in y.split(',')] for y in res['tags'].tolist()]
        
        self.cursor[i] += batch_size
        
        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        if bigram:
            # why * 9 ??
            x = np.zeros([batch_size, maxlen * 9], dtype=np.int32)
            for i, x_i in enumerate(x):
                x_i[:res['length'].values[i] * 9] = words[i]
        else:
            x = np.zeros([batch_size, maxlen], dtype=np.int32)
            for i, x_i in enumerate(x):
                x_i[:res['length'].values[i]] = words[i]
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]
        if classifier is False:
            return x, y, res['length'].values
        else:
            y_class = [round] * batch_size
            return x, y, y_class, res['length'].values
    
    def next_pred_one(self):
        res = self.df.ix[self.pos]
        words = list(map(int, res['words'].split(',')))
        tags = list(map(int, res['tags'].split(',')))
        length = res['length']
        self.pos += 1
        if self.pos == self.total:
            self.pos = 0
        return np.asarray([words], dtype=np.int32), np.asarray([tags], dtype=np.int32), np.asarray([length],
                                                                                                   dtype=np.int32)
    
    def next_all_batch(self, batch_size, bigram=False):
        res = self.df.ix[self.pos: self.pos + batch_size - 1]
        # words = map(lambda x: map(int, x.split(",")), res['words'].tolist())
        # tags = map(lambda x: map(int, x.split(",")), res['tags'].tolist())
        words = [[int(x) for x in y.split(',')] for y in res['words'].tolist()]
        tags = [[int(x) for x in y.split(',')] for y in res['tags'].tolist()]

        self.pos += batch_size
        maxlen = max(res['length'])
        if bigram:
            x = np.zeros([batch_size, maxlen * 9], dtype=np.int32)
            for i, x_i in enumerate(x):
                x_i[:res['length'].values[i] * 9] = words[i]
        else:
            x = np.zeros([batch_size, maxlen], dtype=np.int32)
            for i, x_i in enumerate(x):
                x_i[:res['length'].values[i]] = words[i]
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]
        
        return x, y, res['length'].values
    
    def print_info(self):
        print('dfs shape: ', [len(self.dfs[i]) for i in range(len(self.dfs))])
        print('size: ', self.size)
