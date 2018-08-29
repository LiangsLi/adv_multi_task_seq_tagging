import pandas as pd
import numpy as np


class BucketedDataIterator(object):
    def __init__(self, df, num_buckets=5):
        """
        
        :param df:  dataFrame（pandas.read_csv()）
        :param num_buckets: 桶的数量（近似长度的句子在一个桶里）
        """
        self.df = df
        self.total = len(df)
        # 将语料按照句子长度排序：
        df_sort = df.sort_values('length').reset_index(drop=True)
        # 每个桶的数据数量：
        self.size = self.total // num_buckets
        print("->bucket size: ", self.size)
        self.dfs = []
        for bucket in range(num_buckets - 1):
            # 将数据划分为一个一个桶，保存在 self.dfs 列表中
            self.dfs.append(df_sort.ix[bucket * self.size: (bucket + 1) * self.size - 1])
        #  最后一个桶的数据：
        self.dfs.append(df_sort.ix[(num_buckets - 1) * self.size:])
        self.num_buckets = num_buckets
        
        # cursor[i] will be the cursor for the ith bucket
        # 桶内指针，指向下一个 batch 的起始数据
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
        
        def nextPowerOf2(n):
            count = 0
            # First n in the below
            # condition is for the
            # case where n is 0
            if (n and not (n & (n - 1))):
                return n
            while (n != 0):
                n >>= 1
                count += 1
            return 1 << count
        
        # 如果 batch 的大小超过了桶的最大大小，就重置 batch size 为最接近桶的 size的2的幂数
        if batch_size > self.size:
            vaild_size = nextPowerOf2(self.size)
            if vaild_size > self.size:
                vaild_size = vaild_size // 2
            batch_size = vaild_size
        
        if np.any(self.cursor + batch_size + 1 > self.size):
            # 如果 任何一个桶内 剩余数据不够一个 batch size，就 shuffle 桶，重置桶的指针位置
            # 注意是 任何一个桶！！
            self.epochs += 1
            self.shuffle()
        # 随机选择一个桶!!!
        # 如果需要根据桶的不同改变 batch size 的大小，需要知道使用的哪一个桶
        i = np.random.randint(0, self.num_buckets)
        if i == self.num_buckets - 1:
            batch_size = 16
        else:
            batch_size = int(max(16, batch_size / (2 ** (i // (self.num_buckets // 2)))))
        # 在随机选择的桶中，取出一个 batch 的数据
        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + batch_size - 1]
        # words = map(lambda x: map(int, x.split("||")), res['words'].tolist())
        # tags = map(lambda x: map(int, x.split("||")), res['tags'].tolist())
        # 将 words、tags 转化为 int 数值
        words = [[int(x) for x in y.split(',')] for y in res['words'].tolist()]
        tags = [[int(x) for x in y.split(',')] for y in res['tags'].tolist()]
        # 指针前进
        self.cursor[i] += batch_size
        
        # Pad sequences with 0s so they are all the same length
        # 得到最大长度：
        maxlen = max(res['length'])
        if bigram:
            # why * 9 ??
            x = np.zeros([batch_size, maxlen * 9], dtype=np.int32)
            for i, x_i in enumerate(x):
                x_i[:res['length'].values[i] * 9] = words[i]
        else:
            # padding 数据到最大长度（补0）
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
        # 注意这里不需要担心数组越界的问题,pandas会自己处理
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
