# -*- coding: utf-8 -*-
# Created on 2018/9/3
import pathlib


def fun(file_path):
    def get_begin_end(idxs):
        begin = -1
        end = -1
        B_list = []
        E_list = []
        M_list = []
        for i_, item in enumerate(idxs):
            label = item
            if label == 'B':
                B_list.append(i_)
            elif label == 'E':
                E_list.append(i_)
            # elif label == 'M':
            #     M_list.append(i_)
        if len(B_list) == 0:
            if len(E_list) == 0:
                return -1, -1
            else:
                raise RuntimeError("B!=E")
        elif len(B_list) == len(E_list):
            # begin = B_list[0]
            # end = E_list[0]
            return B_list, E_list
        else:
            # print(line)
            # print('RuntimeError("B!=E 2")')
            return -1, -1
    
    line_num = 0
    all_O_sent_num = 0
    more_than_1_sent_num = 0
    sum_label_len = 0
    with open(file_path, 'r', encoding='utf-8')as f:
        for line in f:
            line_num += 1
            labels = line.strip().split('###')[1].split('||')
            begin, end = get_begin_end(labels)
            if begin == -1:
                all_O_sent_num += 1
            else:
                assert isinstance(begin, list)
                assert isinstance(end, list)
                if len(begin) > 1:
                    more_than_1_sent_num += 1
                elif len(begin) == 0:
                    raise RuntimeError("null list")
                else:
                    for b, e in zip(begin, end):
                        if e < b:
                            raise RuntimeError("e<b")
                        sum_label_len += e - b + 1
    # print("line num:", line_num)
    # print("all O num:", all_O_sent_num)
    print("more than one num:", more_than_1_sent_num,'  ',more_than_1_sent_num/line_num*100,'%')
    # print("ave label len:", sum_label_len / (line_num - all_O_sent_num))


if __name__ == '__main__':
    par_path = pathlib.Path("/Users/liangs/Codes/insurance_data/20_data_v2")
    for file_ in par_path.rglob('test.data'):
        print("\n\n>>>>>",str(file_)[-15:])
        fun(file_)
