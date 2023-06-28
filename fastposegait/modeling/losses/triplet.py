import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, is_hard_loss=False, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.is_hard_loss = is_hard_loss

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        
        if self.is_hard_loss:
            loss_return = self.batch_hard_triplet_loss(ap_dist,an_dist,dist)
        else:
            loss_return = self.batch_all_triplet_loss(ap_dist,an_dist,dist)

        return loss_return, self.info

    def batch_hard_triplet_loss(self,ap_dist,an_dist,dist):
        # largest d(a,p)
        ap_dist = ap_dist.max(-2)[0].view(dist.size(0),dist.size(1)) # [p,n]
        # smallest d(a,n)
        an_dist = an_dist.min(-1)[0].view(dist.size(0),dist.size(1)) #[p,n]
        # [d(a,p)- d(a,n) + margin]+
        loss = F.relu((ap_dist-an_dist + self.margin)) #[p,n]

        hard_loss, _ = self.AvgNonZeroReducer(loss) #[p]

        self.info.update({
            'hard_loss': hard_loss.detach().clone()})
        return hard_loss

    def batch_all_triplet_loss(self,ap_dist,an_dist,dist):
        mean_dist = dist.mean((1, 2))  # [p]
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)
        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})
        return loss_avg
    
    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
