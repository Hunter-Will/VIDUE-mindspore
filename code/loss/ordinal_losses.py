"""
torch
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as O


class OrdinalSupConLoss(nn.Cell):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07):
        super(OrdinalSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def construct(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = O.eye(batch_size, dtype=ms.float32)
        elif labels is not None:
            labels = labels.view(-1, 1)  #b,1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = O.equal(labels, labels.T).float()   #b,b
            labels_r = labels.tile((2,1))
            weights = O.abs((labels_r-labels_r.T)).float()
        else:
            mask = mask.float()

        contrast_count = features.shape[1]  #2
        contrast_feature = O.cat(O.unbind(features, dim=1), axis=0)   #2b, c
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = O.div(
            O.matmul(anchor_feature, contrast_feature.T),
            self.temperature)   #2b,2b
        # for numerical stability
        logits_max, _ = O.max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        # tile mask
        mask = mask.tile((anchor_count, contrast_count))   #2b,2b
        # weights = weights.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = O.eye(mask.shape[0],mask.shape[1])*(-1)+1
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = O.exp(logits) * logits_mask
        exp_logits_sum = O.matmul(exp_logits, weights)
        log_prob = logits - O.log(exp_logits_sum)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
