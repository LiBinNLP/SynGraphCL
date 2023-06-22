#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2023/3/22 16:07
# @Author : libin
# @File : dict_preprocess.py
# @Software: PyCharm

def load_dict(path):
    lines = []
    with open(path, encoding='utf-8') as fin:
        for line in fin.readlines():
            lines.append(line)
        fin.close()

    return lines

def save_dict(path, lines):
    with open(path, encoding='utf-8', mode='w') as fw:
        for line in lines:
            fw.write(line)
        fw.close()

if __name__ == '__main__':
    dicts = [
            ['Person', ['(', ')', ',', ' I', ' II', ' III', ' IV', ' V', ' VI', ' of']],
            ['Organization', ['(', ')', 'Template', ' Category', ' \\', ':']],
            ['Location', ['(', ')', ',']],
            ['Tittle', ['(', ')', '.']],
            ['Corporation', ['(', ')']],
            ['Jobs', ['(', ')']],
        ]

    path = './data/dict/{}.txt'

    for dic in dicts:
        entity_type = dic[0]
        filter_char = dic[1]
        path = path.format(entity_type)
        lines = load_dict(path)
        items = []

        flag = False
        for line in lines:
            flag = True
            for f_char in filter_char:
                if f_char in line:
                    flag = False
                    break
            if flag:
                items.append(line)
        save_dict(path, items)


