import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# -------------------------
# MoCo v2 helper classes
# -------------------------
class ProjectionMLP(nn.Module):
    """Simple 2-layer projection MLP used in MoCo v2: 128 -> 2048 -> 128"""
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        # x: (B, in_dim)
        return self.net(x)

class MoCoWrapper(nn.Module):
    """
    Holds:
      - query_encoder: online encoder + projection head
      - key_encoder: momentum encoder + projection head (no grads)
      - queue: tensor of shape (feat_dim, K)
    """
    def __init__(self, base_encoder, feat_dim=128, K=8192, m=0.999, T=0.2, device='cuda'):
        """
        base_encoder: callable returning (B, feat_dim) already (your PC_Encoder/RD_Encoder)
        We assume base_encoder is an nn.Module (so we can .to(device) and copy)
        """
        super().__init__()
        self.m = m
        self.K = K
        self.T = T
        self.device = device
        self.feat_dim = feat_dim

        # Query encoder and its projection head
        self.encoder_q = base_encoder
        self.proj_q = ProjectionMLP(in_dim=feat_dim).to(device)

        # Key encoder (momentum copy) and its projection head
        # We'll deep-copy the modules so params start equal
        self.encoder_k = copy.deepcopy(base_encoder)
        self.proj_k = copy.deepcopy(self.proj_q)

        # Put key encoder in eval mode and disable grads
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.proj_k.parameters():
            p.requires_grad = False

        # Create the queue (feat_dim x K)
        self.register_buffer("queue", torch.randn(feat_dim, K).to(device))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder: param_k = m * param_k + (1 - m) * param_q"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        keys: (B, feat_dim) - already normalized
        We enqueue along axis K (columns). The queue has shape (feat_dim, K)
        """
        keys = keys.detach()  # (B, feat_dim)
        B = keys.shape[0]

        ptr = int(self.queue_ptr.item())
        # if K not multiple of B, wrap around
        if ptr + B <= self.K:
            self.queue[:, ptr:ptr+B] = keys.T
            ptr = ptr + B
        else:
            # split
            end = self.K - ptr
            self.queue[:, ptr:self.K] = keys[:end].T
            remain = B - end
            self.queue[:, 0:remain] = keys[end:].T
            ptr = remain

        self.queue_ptr[0] = ptr % self.K

    def forward_backbone(self, online_view, online_lengths, target_view, target_lengths, device):
        """
        Compute query features (online_view through encoder_q+proj_q)
                key features   (target_view through encoder_k+proj_k)
        Returns:
            q: (B, feat_dim) normalized
            k: (B, feat_dim) normalized, detached (no grads)
        """
        # online (query)
        q_feat = self.encoder_q(online_view, online_lengths)  # (B, feat_dim)
        q = self.proj_q(q_feat)  # (B, feat_dim)
        q = nn.functional.normalize(q, dim=1)

        # key (momentum) - no gradient
        with torch.no_grad():
            # update momentum encoders BEFORE using them? MoCo does momentum update each step after query encoder grads.
            k_feat = self.encoder_k(target_view, target_lengths)
            k = self.proj_k(k_feat)
            k = nn.functional.normalize(k, dim=1)

        return q, k

    def forward(self, x, lengths):
        """
        For compatibility: just run the query encoder + proj head
        """
        feat = self.encoder_q(x, lengths)  # (B, feat_dim)
        q = self.proj_q(feat)  # (B, feat_dim)
        q = nn.functional.normalize(q, dim=1)
        return q

    def compute_moco_loss(self, q, k):
        """
        q: (B, D), k: (B, D)
        queue: (D, K)
        returns cross-entropy loss
        """
        # positive logits: Nx1
        # negative logits: NxK -> from queue
        D = self.feat_dim
        K = self.K

        # compute positive logits: (B,)
        l_pos = torch.sum(q * k, dim=1, keepdim=True)  # (B,1)
        # compute negative logits: q @ queue  -> (B, K)
        l_neg = torch.matmul(q, self.queue.clone().detach())  # (B, K)

        # logits: concat
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, 1+K)
        logits /= self.T

        # labels: positives are index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)
        return loss