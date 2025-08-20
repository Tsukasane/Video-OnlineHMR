import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class SpatialAwarePooling(nn.Module):
    def __init__(self, patch_grid=(12, 16), embed_dim=1280, out_dim=128, out_h=4, out_w=3):
        super().__init__()
        self.H_patch, self.W_patch = patch_grid
        self.pos_embed = nn.Parameter(torch.randn(1, self.H_patch, self.W_patch, embed_dim))  # [1, H, W, D]

        # 可以用轻量卷积、MLP 或 transformer 做 spatial-aware pooling
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim, kernel_size=3, padding=1), # 混合空间信息，提取最重要的patch
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((out_h, out_w))  # --> [B, out_dim, 1, 1]
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
    def __init__(self, d_model, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        if device is not None:
            self.to(device)

    def forward(self, t: torch.Tensor):  # t: [T]
        t = t[:, None]  # [T, 1]
        return self.mlp(t)  # [T, D]

class SpaceEmbedding(nn.Module):
    def __init__(self, d_model, h=12, w=16, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.h = h
        self.w = w
        if device is not None:
            self.to(device)

    def forward(self, coords: torch.Tensor):  # coords: [H*W, 2]
        assert coords.shape == (self.h * self.w, 2), "Coords shape mismatch"
        return self.mlp(coords)  # [H*W, D]


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_cache, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_cache = max_cache

        # Q/K/V projections for self-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_self = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_self = nn.Linear(hidden_dim, hidden_dim)

        # Q/K/V projections for cross-attention
        self.k_proj_cross = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_cross = nn.Linear(hidden_dim, hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Norm & Dropout
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens, img_feats, causal_mask=None):
        """
        q_tokens: [B, T, D]
        img_feats: [B, T, D] (already pooled & pos added)
        causal_mask: [T, T] or None
        """
        B, T, _ = q_tokens.shape

        # --- Self-Attention ---
        q = self.q_proj(q_tokens)
        k_self = self.k_proj_self(q_tokens)
        v_self = self.v_proj_self(q_tokens)

        q_ = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = k_self.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_self.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_).transpose(1, 2).reshape(B, T, self.hidden_dim)

        # Residual + LN
        out = q_tokens + self.dropout(attn_output)
        out = self.ln1(out)

        # --- Cross-Attention ---
        q_ = self.q_proj(out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_cross = self.k_proj_cross(img_feats).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = self.v_proj_cross(img_feats).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q_, k_cross.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_weights = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_weights, v_cross).transpose(1, 2).reshape(B, T, self.hidden_dim)

        out = out + self.dropout(cross_out)
        out = self.ln2(out)

        # --- FFN ---
        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)
        out = self.ln3(out)

        return out

    @torch.no_grad()
    def incremental_step(self, x_t, mem_t, layer_cache):
        """
        x_t:   [B, 1, D]    当前步的 query token
        mem_t: [B, 1, D]    当前步的图像memory（已pooling+pos，单帧）
        layer_cache: dict {
            'self_k': [B, T_acc, D],
            'self_v': [B, T_acc, D],
            'mem_k':  [B, T_acc, D],
            'mem_v':  [B, T_acc, D],
        }
        返回: x_t_next, layer_cache
        """
        B = x_t.size(0)

        # ---- 计算当前步的 K/V 并拼到缓存（Self）----
        k_self_t = self.k_proj_self(x_t)   # [B,1,D]
        v_self_t = self.v_proj_self(x_t)

        if 'self_k' in layer_cache: # NOTE(yiwen) max_cache=2, only previous and current
            if self.max_cache == layer_cache['self_k'].shape[1]: # FIFO
                layer_cache['self_k'] = layer_cache['self_k'][:,1:,:]
                layer_cache['self_v'] = layer_cache['self_v'][:,1:,:]
            layer_cache['self_k'] = torch.cat([layer_cache['self_k'], k_self_t], dim=1)
            layer_cache['self_v'] = torch.cat([layer_cache['self_v'], v_self_t], dim=1)
        else:
            layer_cache['self_k'] = k_self_t
            layer_cache['self_v'] = v_self_t

        # ---- Self-Attn：q来自当前步，k/v来自缓存 ----
        q_self_t = self.q_proj(x_t)   # [B,1,D]
        q_ = q_self_t.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,1,d]

        k_cached = layer_cache['self_k'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,T,d]
        v_cached = layer_cache['self_v'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_, k_cached.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,h,1,T]
        attn_w = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_w, v_cached)                                            # [B,h,1,d]
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.hidden_dim)

        x_t = self.ln1(x_t + attn_out)  # 残差 + LN（推理时可省dropout）[B,1,D]

        # ---- Cross K/V（来自当前帧 memory），也要缓存 ----
        k_mem_t = self.k_proj_cross(mem_t)  # [B,1,D]
        v_mem_t = self.v_proj_cross(mem_t)

        if 'mem_k' in layer_cache:
            if self.max_cache == layer_cache['mem_k'].shape[1]: # FIFO
                layer_cache['mem_k'] = layer_cache['mem_k'][:,1:,:]
                layer_cache['mem_v'] = layer_cache['mem_v'][:,1:,:]
            layer_cache['mem_k'] = torch.cat([layer_cache['mem_k'], k_mem_t], dim=1)
            layer_cache['mem_v'] = torch.cat([layer_cache['mem_v'], v_mem_t], dim=1)
        else:
            layer_cache['mem_k'] = k_mem_t
            layer_cache['mem_v'] = v_mem_t


        # ---- Cross-Attn：q来自x_t，k/v来自 mem 缓存 ----
        q_cross_t = self.q_proj(x_t)
        q_ = q_cross_t.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_mem = layer_cache['mem_k'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_mem = layer_cache['mem_v'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q_, k_mem.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_w = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_w, v_mem)
        cross_out = cross_out.transpose(1, 2).reshape(B, 1, self.hidden_dim)

        x_t = self.ln2(x_t + cross_out)

        # ---- FFN ----
        ffn_out = self.ffn(x_t)
        x_t = self.ln3(x_t + ffn_out)

        return x_t, layer_cache



class TransformerStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, max_cache, dropout=0.1):
        super().__init__()
        self.pooler = SpatialAwarePooling(patch_grid=(16, 12), embed_dim=input_dim, out_dim=32)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, max_cache, dropout) for _ in range(num_layers)
        ])

    def forward(self, q_tokens, img_feats):
        T = q_tokens.shape[1]
        causal_mask = torch.tril(torch.ones(T, T, device=q_tokens.device)).bool() # NOTE(yiwen) the same causal mask for both self and cross attention
        for layer in self.layers:
            q_tokens = layer(q_tokens, img_feats, causal_mask)
        return q_tokens

    @torch.no_grad()
    def incremental_step(self, x_t, mem_t, cache_layers):
        """
        单步推理：逐层用各自的cache更新
        x_t:          [B, 1, D]
        mem_t:        [B, 1, D]
        cache_layers: list[dict]，长度 = num_layers
        返回: x_t_next, cache_layers
        """
        new_caches = []
        out = x_t
        for i, blk in enumerate(self.layers):
            layer_cache = cache_layers[i] if (cache_layers is not None and i < len(cache_layers)) else {}
            out, layer_cache = blk.incremental_step(out, mem_t, layer_cache)
            new_caches.append(layer_cache)
        return out, new_caches # seperately store the cache for each transformer layer


class SMPLDecoderModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=4, max_frames=400, num_layers=4, max_cache=300, device="cuda"): #input_dim=1280, intermediate_feat_dim=512, hidden_dim=512, num_heads=8, max_frames=300
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_h = 4
        self.out_w = 3
        self.pooler_out_dim = 32
        self.device = device
        self.max_frames = max_frames # upper bound typically for training
        self.max_cache = max_cache # upper bound typically for inference

        self.learned_query = nn.Parameter(torch.randn(1, max_frames, hidden_dim))  # 不同的t index可能有不同的intermediate feature
        self.stack = TransformerStack(input_dim, hidden_dim, num_heads, num_layers, max_cache)

        self.pooler = SpatialAwarePooling(patch_grid=(16, 12), embed_dim=input_dim, out_dim=self.pooler_out_dim, out_h=self.out_h, out_w=self.out_w)

        self.pose_head = nn.Linear(hidden_dim, 24 * 6)
        self.shape_head = nn.Linear(hidden_dim, 10)
        self.cam_head = nn.Linear(hidden_dim, 3)

        self.timepos_encoder = ContinuousTimeEmbedding(d_model=input_dim, device=self.device)
        self.spacepos_encoder = SpaceEmbedding(d_model=input_dim, device=self.device)

        self.input_proj = nn.Linear(self.out_h*self.out_w*self.pooler_out_dim, hidden_dim)

    def forward(self, img_feats_all, q_tokens=None):

        img_feats_all = self.add_pos_to_seqtokens(img_feats_all, self.device) # NOTE(yiwen) check this
        img_feats_all = self.pooler(img_feats_all)  # NOTE(yiwen) compress frame info to representative patch 某一帧的压缩版重要信息在过往帧中的响应(temporal)
        
        B, T, _ = img_feats_all.shape
        if q_tokens is None:
            q_tokens = self.learned_query[:, :T, :].expand(B, T, -1)

        img_feats_all = self.input_proj(img_feats_all)

        out = self.stack(q_tokens, img_feats_all)
        pose = self.pose_head(out)
        shape = self.shape_head(out)
        cam = self.cam_head(out)

        return pose, shape, cam

    def add_pos_to_seqtokens(self, x, device, t=None):
        """
        x: [B, T, N_patch, D]
        Seperately encode time and space
        """
        B, T, N_patch, D = x.shape

        H = 16
        W = 12
        assert H * W == N_patch, f"cannot be reshape to meshgrid"

        if T==1: # single frame inference
            time_ids = torch.tensor([1.0], device=device) # always corresponds to the last frame in cache sequence
        else:
            time_ids = torch.linspace(0, 1, T, device=device)  # [T]
        time_pos = self.timepos_encoder(time_ids)

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1).float()  # [H, W, 2]
        coords = coords / torch.tensor([H, W], device=device).float()  # normalize
        coords = coords.view(-1, 2)  # [N_patch, 2]
        space_pos = self.spacepos_encoder(coords)  # [N_patch, D]

        x = x + time_pos[None, :, None, :] + space_pos[None, None, :, :]  # time(copy for same space) space(copy for same time)

        return x

    @torch.no_grad()
    def inference_step(self, img_feat_t, q_tokens=None, cache=None, t=None, device="cpu"):
        """
        img_feat_t: [B, 1, N_patch, D]（单帧）
        cache: {
            'layers': [  # len == num_layers
                {'self_k':..., 'self_v':..., 'mem_k':..., 'mem_v':...},
                ...
            ]
        }
        """
        # 先加Pos，再pool
        img_feat_t = self.add_pos_to_seqtokens(img_feat_t, device, t)  # [B,1,Np,D]
        img_feat_t = self.pooler(img_feat_t)                        # [B,1,D]
        B = img_feat_t.size(0)

        if q_tokens is None:
            assert t is not None, "inference_step requires t or explicit q_tokens"
            if t >= self.max_cache:
                t = self.max_cache - 1 # NOTE(yiwen) relative position in cache
            q_tokens = self.learned_query[:, t:t+1, :].expand(B, 1, -1)   # [B,1,D]

        img_feat_t = self.input_proj(img_feat_t)

        cache_layers = cache['layers'] if (cache is not None and 'layers' in cache) else None
        out, new_cache_layers = self.stack.incremental_step(q_tokens, img_feat_t, cache_layers)

        pose  = self.pose_head(out)     # [B,1,24*6]
        shape = self.shape_head(out)    # [B,1,10]
        cam   = self.cam_head(out)      # [B,1,3]

        new_cache = {'layers': new_cache_layers}
        return pose, shape, cam, new_cache



if __name__=="__main__":

    train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # decoder = KVCacheDecoder(intermediate_feat_dim=384, hidden_dim=512).to(device)
    decoder = SMPLDecoderModel(hidden_dim=512).to(device)

    # NOTE(yiwen) init dummy input  batch=2, seq_len=4, 1280 feats channels, 64x48 resolution
    B, T, D, H, W = 2, 250, 1280, 16, 12 # if not using bbox 3 dim here.
    N_patch = H*W
    x = torch.randn(B, T, H*W, D).to(device)


    if train:
        print(f"input shape: {x.shape}")  # [B, T, N_patch, D] = [2, 4*192, 1280]
        smpl_pose, smpl_shape, smpl_cam = decoder(img_feats_all=x, q_tokens=None)
    else:
        cache = None
        dummy_inferenceinput = x[:, 0:1, :, :]
        print(f"input shape: {dummy_inferenceinput.shape}")  # [B, T, N_patch, D] = [2, 4*192, 1280]
        inference_seqlen = 200

        for t in range(inference_seqlen): # NOTE(yiwen) test here again
            smpl_pose, smpl_shape, smpl_cam, cache = decoder.inference_step(img_feat_t=dummy_inferenceinput, t=t, device=device, cache=cache)

    print("smpl pose shape:", smpl_pose.shape)    # B, T, 144
    print("smpl shape shape:", smpl_shape.shape)    #, B, T, 10
    print("smpl cam shape:", smpl_cam.shape)    # B, T, 3
