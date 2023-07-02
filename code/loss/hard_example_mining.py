import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O


class HEM(nn.Cell):
    def __init__(self, hard_thre_p=0.5, random_thre_p=0.1):
        super(HEM, self).__init__()
        self.hard_thre_p = hard_thre_p
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()

    def hard_mining_mask(self, x, y):
        b, c, h, w = x.size()

        hard_mask = O.zeros((b, 1, h, w))
        res = O.sum(O.abs(x - y), dim=1, keepdim=True)
        res_line = res.view(b, -1)
        res_sort = [res_line[i].sort(descending=True) for i in range(b)]
        hard_thre_ind = int(self.hard_thre_p * w * h)
        for i in range(b):
            thre_res = res_sort[i][0][hard_thre_ind].item()
            hard_mask[i] = (res[i] > thre_res).astype(ms.float32)

        random_thre_ind = int(self.random_thre_p * w * h)
        random_mask = O.zeros((b, 1 * h * w))
        for i in range(b):
            random_mask[i, :random_thre_ind] = 1.
            O.shuffle(random_mask[i])
        random_mask = O.reshape(random_mask, (b, 1, h, w))

        mask = hard_mask + random_mask
        mask = (mask > 0.).astype(ms.float32)

        return mask

    def construct(self, x, y):
        mask = self.hard_mining_mask(x, y)

        hem_loss = self.L1_loss(x * mask, y * mask)

        return hem_loss
