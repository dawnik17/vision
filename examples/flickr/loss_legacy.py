"""
https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
"""

from torch.nn.functional import normalize
import torch.nn as nn
import torch
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, temperature, device, reduction="mean"):
        super(ClipLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.reduction = reduction
        
    def forward(self, image_embeddings, text_embeddings):
        """
        image_embedding and text_embedding shape = [batch, emb_dim]
        """
        images_similarity = image_embeddings @ text_embeddings.T / self.temperature
        texts_similarity = images_similarity.T

        batch = images_similarity.shape[0]
        labels = torch.arange(batch).long().to(self.device)

        total_loss = (
            F.cross_entropy(images_similarity, labels, reduction="none") + 
            F.cross_entropy(texts_similarity, labels, reduction="none")
        )

        if self.reduction is None:
            return total_loss

        elif self.reduction == "sum":
            return total_loss.sum()

        else:
            return total_loss.sum() / (2 * batch)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, temperature=1, reduction="mean", margin=0):
        super(CosineSimilarityLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.margin = margin

    def forward(self, image_embeddings, text_embeddings):
        """
        image_embedding and text_embedding shape = [batch, emb_dim]
        """
        # cosine similarity
        image_embeddings = normalize(image_embeddings)
        text_embeddings = normalize(text_embeddings)

        images_similarity = image_embeddings @ text_embeddings.T / self.temperature
        batch = images_similarity.shape[0]

        # for the diagonal (positive score)
        fill_tensor = torch.ones(batch, batch, device=image_embeddings.device)
        fill_tensor.fill_diagonal_(-1)
        positive_score = fill_tensor * (
            images_similarity - torch.eye(batch, device=image_embeddings.device)
        )

        # for non diagonal elements (negative score)
        fill_tensor = torch.zeros(batch, batch, device=image_embeddings.device)
        fill_tensor.fill_diagonal_(-1e4)

        margin_tensor = torch.full((batch, batch), self.margin, device=image_embeddings.device)
        margin_tensor.fill_diagonal_(0)

        final_score = torch.max(fill_tensor, positive_score - margin_tensor)

        return final_score.sum() if self.reduction == "sum" else final_score.mean()


class HingeLoss(nn.Module):
    def __init__(self, temperature, delta, reduction="mean"):
        super(HingeLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        # cosine similarity
        image_embeddings = normalize(image_embeddings)
        text_embeddings = normalize(text_embeddings)

        images_similarity = image_embeddings @ text_embeddings.T / self.temperature
        batch = images_similarity.shape[0]
        device = images_similarity.device

        # scoring
        score = (
            (torch.eye(batch, device=device) * self.delta)
            - (self.delta - images_similarity)
        ) * (
            2 * torch.eye(batch, device=device)
            - torch.ones(batch, batch, device=device)
        )

        fill_tensor = torch.zeros(batch, batch, device=device)
        fill_tensor.fill_diagonal_(-1e4)

        final_score = torch.max(fill_tensor, score)
        return final_score.sum() if self.reduction == "sum" else final_score.mean()


class BCELoss(nn.Module):
    def __init__(self, reduction, temperature):
        super(BCELoss, self).__init__()
        # self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        self.criterion = nn.BCELoss(reduction=reduction)
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        image_embeddings = normalize(image_embeddings)
        text_embeddings = normalize(text_embeddings)

        batch = image_embeddings.shape[0]
        target = torch.eye(batch, device=image_embeddings.device)
        score = image_embeddings @ text_embeddings.T / self.temperature

        return self.criterion(score, target)


# cross entropy + pair-wise loss
class CEPWLoss(nn.Module):
    def __init__(self, temperature, device):
        super(CEPWLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, image_embeddings, text_embeddings):
        """
        image_embedding and text_embedding shape = [batch, emb_dim]
        """
        images_similarity = image_embeddings @ text_embeddings.T / self.temperature
        texts_similarity = images_similarity.T

        batch = images_similarity.shape[0]

        # Cross Entropy Loss
        ce_target = torch.arange(batch).long().to(self.device)

        total_ce_loss = (
            F.cross_entropy(images_similarity, ce_target, reduction="mean") + 
            F.cross_entropy(texts_similarity, ce_target, reduction="mean")
        )

        # Pair-wise BCE Loss
        base = torch.diag(images_similarity).unsqueeze(-1)
        
        image_pair = F.sigmoid(base - images_similarity)
        image_pair = image_pair[~torch.eye(batch, dtype=bool)]

        text_pair = F.sigmoid(base - texts_similarity)
        text_pair = text_pair[~torch.eye(batch, dtype=bool)]
        
        bce_target = torch.ones(batch * (batch - 1), device=self.device).float()
        
        total_bce_loss = (
            F.binary_cross_entropy(image_pair, bce_target, reduction="mean") +
            F.binary_cross_entropy(text_pair, bce_target, reduction="mean")
        )

        # total loss
        total_loss = total_ce_loss + total_bce_loss
        
        return {"total_loss": total_loss / 4,
                "cross_entropy_loss": total_ce_loss / 2,
                "bce_loss": total_bce_loss / 2}
