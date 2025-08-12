import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class SpatialAwarePooling(nn.Module):
    def __init__(self, patch_grid=(12, 16), embed_dim=1280, out_dim=128):
        super().__init__()
        self.H_patch, self.W_patch = patch_grid
        self.pos_embed = nn.Parameter(torch.randn(1, self.H_patch, self.W_patch, embed_dim))  # [1, H, W, D]

        self.out_h = 4
        self.out_w = 3
        # 可以用轻量卷积、MLP 或 transformer 做 spatial-aware pooling
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim, kernel_size=3, padding=1), # 混合空间信息，提取最重要的patch
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.out_h, self.out_w))  # --> [B, out_dim, 1, 1]
        )

    def forward(self, feat):  # [B, T, N_patch, D]
        B, T, N_patch, D = feat.shape
        assert N_patch == self.H_patch * self.W_patch, "Patch size mismatch"

        feat = feat.view(B*T, self.H_patch, self.W_patch, D)  # [B*T, H, W, D]
        feat = feat + self.pos_embed

        feat = feat.permute(0, 3, 1, 2)  # [B*T, D, H, W]
        pooled = self.spatial_pool(feat)  # [B*T, out_dim, out_h, out_w]

        # compress the channel dimension, also compress the spatial feature
        pooled = pooled.reshape(B, T, -1)   # shift spatial feature to channel [B, T, out_dim*out_h*out_w]
        return pooled


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor):  # t: [T]
        t = t[:, None]  # [T, 1]
        return self.mlp(t)  # [T, D]

class SpaceEmbedding(nn.Module):
    def __init__(self, d_model, h=12, w=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.h = h
        self.w = w

    def forward(self, coords: torch.Tensor):  # coords: [H*W, 2]
        assert coords.shape == (self.h * self.w, 2), "Coords shape mismatch"
        return self.mlp(coords)  # [H*W, D]


class KVCacheDecoder(nn.Module):
    def __init__(self, input_dim=1280, intermediate_feat_dim=512, hidden_dim=512, num_heads=8, max_frames=300): 
        # TODO(yiwen) the max_frames cannot be too small, need further adapt to sliding window for cache
        # TODO(yiwen) also need to change the dataloader

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.pooler = SpatialAwarePooling(patch_grid=(16, 12), embed_dim=input_dim, out_dim=32)

        # projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_self = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_self = nn.Linear(hidden_dim, hidden_dim)

        self.k_proj_cross = nn.Linear(intermediate_feat_dim, hidden_dim) # memory
        self.v_proj_cross = nn.Linear(intermediate_feat_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # learnable query tokens (used if none provided)
        self.learned_query = nn.Parameter(torch.randn(1, max_frames, hidden_dim))

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # heads
        self.pose_head = nn.Linear(hidden_dim, 24 * 6)
        self.shape_head = nn.Linear(hidden_dim, 10)
        self.cam_head = nn.Linear(hidden_dim, 3)

    def forward(self, img_feats_all, q_tokens=None, device="cpu"):
        """
        Training-time forward: use full sequence with causal masking.
        img_feats_all: [B, T, D_img]
        q_tokens:      [B, T, D] (optional); if None, use learned_query[:T]
        Returns: pose, shape, cam, shape = [B, T, ...]
        """
        img_feats_all = self.add_pos_to_seqtokens(img_feats_all, device)
        
        # NOTE(yiwen) compress frame info to representative patch 某一帧的压缩版重要信息在过往帧中的响应(temporal)
        img_feats_all = self.pooler(img_feats_all)

        B, T, _ = img_feats_all.shape
        
        if q_tokens is None:
            q_tokens = self.learned_query[:, :T, :].expand(B, T, -1)  # copy batch times --> [B, T, D]

        q = self.q_proj(q_tokens)
        k_self = self.k_proj_self(q_tokens)
        v_self = self.v_proj_self(q_tokens)
        k_cross = self.k_proj_cross(img_feats_all)
        v_cross = self.v_proj_cross(img_feats_all)
 

        # --- Self-Attention with causal mask ---
        q_ = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        k_ = k_self.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_self.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, T, T]

        causal_mask = torch.tril(torch.ones(T, T, device=q.device)).bool()
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_)  # [B, h, T, d]
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_dim)

        out = attn_output + q  # residual

        # --- Cross-Attention ---
        q_ = self.q_proj(out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = k_cross.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_cross.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_weights = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_weights, v_)
        cross_out = cross_out.transpose(1, 2).reshape(B, T, self.hidden_dim)

        out = out + cross_out
        out = out + self.ffn(out)

        pose = self.pose_head(out)    # [B, T, 24*6]
        shape = self.shape_head(out)  # [B, T, 10]
        cam = self.cam_head(out)      # [B, T, 3]

        return pose, shape, cam

    def inference_step(self, q_tokens=None, img_feat_t=None, cache=None, t=None, device="cpu"):
        """
        q_tokens: [B, 1, D]  - current query token
        img_feat_t: [B, 1, D_img] - image feature of current frame (1-step)
        cache: {
            'self_k': [B, T_accumulate, D],
            'self_v': [B, T_accumulate, D],
            'mem_k': [B, T_accumulate, D],
            'mem_v': [B, T_accumulate, D]
        }
        t: current time index (used for learned query)
        """

        img_feat_t = self.add_pos_to_seqtokens(img_feat_t, device) # NOTE(yiwen) check this
        
        # NOTE(yiwen) compress frame info to representative patch 某一帧的压缩版重要信息在过往帧中的响应(temporal)
        img_feat_t = self.pooler(img_feat_t)
        B = img_feat_t.size(0)

        if q_tokens is None:
            assert t is not None
            q_tokens = self.learned_query[:, t:t+1, :].expand(B, 1, -1)  # [B, 1, D] the first clue for all batches

        q = self.q_proj(q_tokens)              # [B, 1, D]
        k_self = self.k_proj_self(q_tokens)    # [B, 1, D]
        v_self = self.v_proj_self(q_tokens)

        k_cross = self.k_proj_cross(img_feat_t) 
        v_cross = self.v_proj_cross(img_feat_t)

        if cache is None:
            cache = {}

        # ----- Update cache -----
        if 'self_k' in cache:
            cache['self_k'] = torch.cat([cache['self_k'], k_self], dim=1)  # [B, T+1, D]
            cache['self_v'] = torch.cat([cache['self_v'], v_self], dim=1)
        else:
            cache['self_k'] = k_self
            cache['self_v'] = v_self

        if 'mem_k' in cache:
            cache['mem_k'] = torch.cat([cache['mem_k'], k_cross], dim=1)
            cache['mem_v'] = torch.cat([cache['mem_v'], v_cross], dim=1)
        else:
            cache['mem_k'] = k_cross
            cache['mem_v'] = v_cross

        # ----- Self-Attention (causal) -----
        q_ = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)      # [B, h, 1, d]
        k_ = cache['self_k'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        v_ = cache['self_v'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, 1, T]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_)  # [B, h, 1, d] same as q shape
        attn_output = attn_output.transpose(1, 2).reshape(B, 1, self.hidden_dim)  # [B, 1, D=h*d]

        out = attn_output + q  # residual

        # ----- Cross-Attention (on memory) -----
        q_ = self.q_proj(out).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = cache['mem_k'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = cache['mem_v'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        cross_attn_weights = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_attn_weights = F.softmax(cross_attn_weights, dim=-1)
        cross_output = torch.matmul(cross_attn_weights, v_)
        cross_output = cross_output.transpose(1, 2).reshape(B, 1, self.hidden_dim)

        out = out + cross_output  # residual

        # ----- Feed-Forward -----
        out = out + self.ffn(out)

        # ----- Output Heads -----
        pose = self.pose_head(out)    # [B, 1, 24*6]
        shape = self.shape_head(out)  # [B, 1, 10]
        cam = self.cam_head(out)      # [B, 1, 3]

        return pose, shape, cam, cache


    def add_pos_to_seqtokens(self, x, device):
        """
        Seperately encode time and space
        """
        B, T, N_patch, D = x.shape
        H, W = 16, 12
        time_ids = torch.linspace(0, 1, T).to(x.device)  # Normalize time [0,1]
        timepos_encoder = ContinuousTimeEmbedding(d_model=D).to(device)  # [T, D]
        spacepos_encoder = SpaceEmbedding(d_model=D, h=H, w=W).to(device)
        time_pos = timepos_encoder(time_ids)  # [T, D]

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')  # [12, 16]
        coords = torch.stack([grid_y, grid_x], dim=-1).float()  # [12, 16, 2]
        coords = coords / torch.tensor([H, W])  # 归一化
        coords = coords.view(-1, 2).to(x.device)  # [192, 2]
        space_pos = spacepos_encoder(coords)  # MLP(2→D)，输出 shape [192, D]
        
        x = x + time_pos[None, :, None, :] + space_pos[None, None, :, :]
        seq_feat = x.view(B, T, N_patch, D)

        return seq_feat  # [B, T, N_patch, D]


if __name__=="__main__":

    train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = KVCacheDecoder(intermediate_feat_dim=384, hidden_dim=512).to(device)

    # NOTE(yiwen) init dummy input  batch=2, seq_len=4, 1280 feats channels, 64x48 resolution
    B, T, D, H, W = 2, 250, 1280, 16, 12 # if not using bbox 3 dim here.
    N_patch = H*W
    x = torch.randn(B, T, H*W, D).to(device)

    dummy_inferenceinput = x[:, 0:1, :, :]

    print(f"input shape: {x.shape}")  # [B, T, N_patch, D] = [2, 4*192, 1280]

    if train:
        smpl_pose, smpl_shape, smpl_cam = decoder(img_feats_all=x, q_tokens=None, device=device)
    else:
        cache = None
        for t in range(inference_seqlen): # NOTE(yiwen) test here again
            smpl_pose, smpl_shape, smpl_cam, cache = decoder.inference_step(img_feat_t=dummy_inferenceinput, t=t, device=device, cache=cache)

    print("smpl pose shape:", smpl_pose.shape)    # B, T, 144
    print("smpl shape shape:", smpl_shape.shape)    #, B, T, 10
    print("smpl cam shape:", smpl_cam.shape)    # B, T, 3
