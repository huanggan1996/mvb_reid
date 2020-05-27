import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        super().__init__(self)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)  # 构建一个默认value为list的字典
        for index, (_, bagid, _) in enumerate(data_source):
            self.index_dic[bagid].append(index)
        self.bagids = list(self.index_dic.keys())
        self.num_identities = len(self.bagids)

    def __iter__(self):  # 这个是用来产生迭代索引值的，也就是指定每个step需要读取哪些数据
        indices = torch.randperm(self.num_identities)  # 给定参数n，返回一个从0到n-1的随机整数排列
        ret = []
        for i in indices:
            bagid = self.bagids[i]
            t = self.index_dic[bagid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            # 从t中取size个值，replace:True表示可以取相同数字，False表示不可以取相同数字
            ret.extend(t)
        return iter(ret)

    def __len__(self):  # 这个是用来返回每次迭代器的长度
        return self.num_identities * self.num_instances
