import torch
import torch.nn as nn
import torch.nn.functional as F

class IS_TripletLoss(nn.Module):
    def __init__(self, margin=1.0, K=5):
        super(IS_TripletLoss, self).__init__()
        self.margin = margin
        self.K = K

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        triplet_loss = F.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(triplet_loss)
        
