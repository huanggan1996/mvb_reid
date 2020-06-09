import numpy as np
import torch
from config import cfg
from models.network import BagReID
from utils.re_ranking import re_ranking as re_ranking_func
from utils.load_data import build_data_loader


class Evaluator:
    def __init__(self, model):
        self.model = model.cuda() if cfg.CUDA else model

    def evaluate(self, queryloader, galleryloader, re_ranking=False):
        self.model.eval()
        qf = []
        imgs_id = []
        imgs_true_id= []
        for inputs in queryloader:
            img, img_id, true_id = self.parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self.forward(img)
            feature_hflip = self.forward(img_hflip)
            feature_vflip = self.forward(img_vflip)
            feature_hvflip = self.forward(img_hvflip)
            qf.append(torch.max(feature, torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            imgs_id.extend(img_id)
            imgs_true_id.extend(true_id)
        qf = torch.cat(qf, 0)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf = []
        g_bagids = []
        for inputs in galleryloader:
            img, bagid, _ = self.parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self.forward(img)
            feature_hflip = self.forward(img_hflip)
            feature_vflip = self.forward(img_vflip)
            feature_hvflip = self.forward(img_hvflip)
            gf.append(torch.max(feature, torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            g_bagids.extend(bagid)
        gf = torch.cat(gf, 0)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1,  -2, qf, gf.t())  # a2+b2-2ab

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist, k1=5, k2=5, lambda_value=0.3))
        else:
            distmat = q_g_dist

        self.to_csv(distmat, imgs_id, g_bagids, imgs_true_id)

    def parse_data(self, inputs):
        imgs, bag_ids, camids = inputs
        if cfg.CUDA:
            imgs = imgs.cuda()
        return imgs, bag_ids, camids

    def forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def flip_horizontal(self, image):
        '''flip horizontal'''
        inv_idx = torch.arange(image.size(3) - 1, -1, -1)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(3, inv_idx)
        return img_flip

    def flip_vertical(self, image):
        '''flip vertical'''
        inv_idx = torch.arange(image.size(2) - 1, -1, -1)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(2, inv_idx)
        return img_flip

    def to_csv(self, distmat, imgs_id, g_bagids, gt_id):
        rank = torch.argsort(distmat, dim=1)
        pre_id = []
        ret = ''
        with open(cfg.TEST.OUTPUT, 'w') as f:
            for ii, row in enumerate(rank):
                line = ''
                img_id = imgs_id[ii]
                img_id = '{:05d}'.format(int(img_id)) + ','
                line += img_id
                bag_set = set()
                for jj, col in enumerate(row):
                    bagid = int(g_bagids[col])
                    if bagid not in bag_set:
                        score = distmat[ii, col]
                        line += '{:04d}'.format(bagid) + ',' + '{:.8f}'.format(score) + ','
                        bag_set.add(bagid)
                line = line + gt_id[ii] + '\n'
                ret += line
                pre_id.append(g_bagids[row[0]])
            f.write(ret)
        print("Rank 1: {:.2f} % ".format(
            np.sum([pre_id[i] == gt_id[i] for i in range(len(gt_id))]) / len(gt_id) * 100))


if __name__ == '__main__':
    dataset, _, query_loader, gallery_loader = build_data_loader()
    model = BagReID(dataset.num_train_bags)
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, re_ranking=True)