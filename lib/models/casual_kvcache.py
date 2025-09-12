import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import einops

from .components.pose_transformer import TransformerDecoder

import numpy as np
from skimage.util.shape import view_as_windows


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
        self.q_proj_self = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_self = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_self = nn.Linear(hidden_dim, hidden_dim)

        # Q/K/V projections for cross-attention
        self.q_proj_cross = nn.Linear(hidden_dim, hidden_dim)
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

    def forward(self, q_tokens, img_feats):
        """
        q_tokens: [B, h*w, D]
        img_feats: [B, t*h*w, D] (already pooled & pos added)
        """
        B, T1, _ = q_tokens.shape
        _, T2, _ = img_feats.shape

        # --- Self-Attention ---
        q = self.q_proj_self(q_tokens)
        k_self = self.k_proj_self(q_tokens)
        v_self = self.v_proj_self(q_tokens)

        q_ = q.view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        k_ = k_self.view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        v_ = v_self.view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_).transpose(1, 2).reshape(B, T1, self.hidden_dim)

        # Residual + LN
        out = q_tokens + self.dropout(attn_output)
        out = self.ln1(out)

        # --- Cross-Attention ---
        q_ = self.q_proj_cross(out).view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        k_cross = self.k_proj_cross(img_feats).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = self.v_proj_cross(img_feats).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q_, k_cross.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_weights = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_weights, v_cross).transpose(1, 2).reshape(B, T1, self.hidden_dim)

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
        x_t:   [B, h*w, D]    current step query token
        mem_t: [B, t*h*w, D]    previous steps k, v
        layer_cache: dict {
            'mem_k':  [B, T_mem, D],
            'mem_v':  [B, T_mem, D],
        }
        Return: x_t (fused self spatial info + previous temporal info), layer_cache
        """
        B = x_t.shape[0]
        T = x_t.shape[1] # t*h*w

        k_self_t = self.k_proj_self(x_t)   # [B,1,D]
        v_self_t = self.v_proj_self(x_t)

        # self atten
        q_self_t = self.q_proj_self(x_t)   # [B,T,D]
        q_ = q_self_t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,T(thw),d]

        k_self_t = k_self_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,T,d]
        v_self_t = v_self_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_, k_self_t.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,h,1,T]
        attn_w = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_w, v_self_t)                                            # [B,h,1,d]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        x_t = self.ln1(x_t + attn_out)  # residual + LN（no dropout in inference）[B,T,D]

        # cross atten
        k_mem_t = self.k_proj_cross(mem_t)  # [B,T,D]
        v_mem_t = self.v_proj_cross(mem_t)

        q_cross_t = self.q_proj_cross(x_t)
        q_ = q_cross_t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if 'mem_k' in layer_cache:
            k_mem = layer_cache['mem_k'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_mem = layer_cache['mem_v'].view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k_mem = k_mem_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_mem = v_mem_t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q_, k_mem.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_w = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_w, v_mem)
        cross_out = cross_out.transpose(1, 2).reshape(B, T, self.hidden_dim)

        x_t = self.ln2(x_t + cross_out) # 2, 192, 512

        # cache
        if 'mem_k' in layer_cache:
            if self.max_cache - layer_cache['mem_k'].shape[1] < T: # FIFO
                layer_cache['mem_k'] = layer_cache['mem_k'][:,T:,:]
                layer_cache['mem_v'] = layer_cache['mem_v'][:,T:,:]
            layer_cache['mem_k'] = torch.cat([layer_cache['mem_k'], k_mem_t], dim=1)
            layer_cache['mem_v'] = torch.cat([layer_cache['mem_v'], v_mem_t], dim=1)
        else:
            layer_cache['mem_k'] = k_mem_t
            layer_cache['mem_v'] = v_mem_t

        # ---- FFN ----
        ffn_out = self.ffn(x_t)
        x_t = self.ln3(x_t + ffn_out)

        return x_t, layer_cache



class TransformerStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, max_cache, dropout=0.1):
        super().__init__()
        # make multiple transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, max_cache, dropout) for _ in range(num_layers)
        ])

    def forward(self, q_tokens, img_feats):
        T = q_tokens.shape[1]
        for layer in self.layers:
            q_tokens = layer(q_tokens, img_feats)
        return q_tokens

    @torch.no_grad()
    def incremental_step(self, x_t, mem_t, cache_layers):
        """
        one step inference, update cache
        x_t:          [B, 1, D]
        mem_t:        [B, 1, D]
        cache_layers: list[dict], length = num_layers
        Return: x_t, cache_layers
        """
        new_caches = []
        out = x_t
        for i, blk in enumerate(self.layers):
            layer_cache = cache_layers[i] if (cache_layers is not None and i < len(cache_layers)) else {}
            out, layer_cache = blk.incremental_step(out, mem_t, layer_cache)
            new_caches.append(layer_cache)
        return out, new_caches # seperately store the cache for each transformer layer


class SMPLTransformerDecoderHead(nn.Module): # use the context from one image
    """ HMR2 Cross-attention based SMPL Transformer decoder
    """
    def __init__(self, ):
        super().__init__()
        transformer_args = dict(
            depth = 6,  # originally 6
            heads = 8,
            mlp_dim = 1024,
            dim_head = 64,
            dropout = 0.0,
            emb_dropout = 0.0,
            norm = "layer",
            context_dim = 512,
            num_tokens = 1,
            token_dim = 1,
            dim = 1024
            )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = 1024
        npose = 24*6
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load('/ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/data/smpl/smpl_mean_params.npz')
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        
    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # Pass through transformer
        token = torch.zeros(batch_size, 1, 1).to(x.device)
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        # Readout from token_out
        pred_pose = self.decpose(token_out)  + init_body_pose
        pred_shape = self.decshape(token_out)  + init_betas
        pred_cam = self.deccam(token_out)  + init_cam

        return pred_pose, pred_shape, pred_cam


class SMPLDecoderModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=4, num_layers=4, max_cache=300, device="cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.max_cache = max_cache # upper bound typically for inference

        self.stack = TransformerStack(input_dim, hidden_dim, num_heads, num_layers, max_cache)
        self.smpl_head = SMPLTransformerDecoderHead()

        self.timepos_encoder = ContinuousTimeEmbedding(d_model=input_dim, device=self.device)
        self.spacepos_encoder = SpaceEmbedding(d_model=input_dim, device=self.device)

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)

    def forward(self, img_feats_all, q_tokens=None):

        img_feats_all = self.add_pos_to_seqtokens(img_feats_all, self.device) # NOTE(yiwen) check this
        batch_size = img_feats_all.shape[0]

        img_feats_all = einops.rearrange(img_feats_all, 'b t (h w) c -> b (t h w) c', b=batch_size, h=16, w=12) # b*h*w, 
        
        B, T, _ = img_feats_all.shape
        q_tokens = self.input_proj(q_tokens)
        img_feats_all = self.input_proj(img_feats_all) # B, T, 512

        # the stack of transformer lys
        out = self.stack(q_tokens, img_feats_all) # B, T, 512
        # get the image level feature
        out = einops.rearrange(out, 'b (t h w) c -> (b t) c h w', b=batch_size, h=16, w=12) 
        pose, shape, cam = self.smpl_head(out)


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
        img_feat_t: [B, 1, N_patch, D]（single frame）
        cache: {
            'layers': [  # len == num_layers
                {'self_k':..., 'self_v':..., 'mem_k':..., 'mem_v':...},
                ...
            ]
        }
        """
        img_feat_t = self.add_pos_to_seqtokens(img_feat_t, device, t)  # [B,1,Np,D]
        batch_size = img_feat_t.shape[0]
        img_feat_t = einops.rearrange(img_feat_t, 'b t (h w) c -> b (t h w) c', b=batch_size, h=16, w=12)
        B = img_feat_t.size(0) # we have a q for each patch this time

        q_tokens = self.input_proj(q_tokens) # 2, 192, 512
        img_feat_t = self.input_proj(img_feat_t) # 2, 192, 512
        cache_layers = cache['layers'] if (cache is not None and 'layers' in cache) else None
        out, new_cache_layers = self.stack.incremental_step(q_tokens, img_feat_t, cache_layers)
        out = einops.rearrange(out, 'b (t h w) c -> (b t) c h w', b=batch_size, h=16, w=12)

        pose, shape, cam = self.smpl_head(out)

        new_cache = {'layers': new_cache_layers}
        return pose, shape, cam, new_cache



if __name__=="__main__":


    train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE(yiwen) init dummy input  batch=2, seq_len=4, 1280 feats channels, 64x48 resolution
    B, T, D, H, W = 24, 16, 1280, 16, 12 # if not using bbox 3 dim here.
    N_patch = H*W
    x = torch.randn(B, T, N_patch, D).to(device)
    
    max_memt = 2
    max_cache = max_memt * H * W

    decoder = SMPLDecoderModel(hidden_dim=512, max_cache=max_cache).to(device)

    if train:
        print(f"input shape: {x.shape}")  # [B, T, N_patch, D] = [2, 4*192, 1280]
        
        total_num = x.shape[1]
        chunks = x.unfold(dimension=1, size=max_memt+1, step=1)
        chunks = chunks.reshape(-1,*chunks.shape[2:]).permute(0,3,1,2) # B*N, window_length, h*w, D
        q_token_raw = chunks[:,2:3,...] # current
        img_feats_raw = chunks[:,0:2,...] # previous

        new_bs = img_feats_raw.shape[0]
        
        q_token2 = einops.rearrange(q_token_raw, 'b t (h w) c -> b (t h w) c', b=new_bs, h=16, w=12)
        smpl_pose, smpl_shape, smpl_cam = decoder(img_feats_all=img_feats_raw, q_tokens=q_token2)
        
    else:
        cache = None
        dummy_inferenceinput = x[:, 0:1, :, :]
        print(f"input shape: {dummy_inferenceinput.shape}")  # [B, T, N_patch, D] = 2, 1, 192, 1280
        inference_seqlen = 200

        for t in range(inference_seqlen):
            q_token2 = einops.rearrange(dummy_inferenceinput, 'b t (h w) c -> b (t h w) c', b=B, h=16, w=12)
            smpl_pose, smpl_shape, smpl_cam, cache = decoder.inference_step(img_feat_t=dummy_inferenceinput, 
                                                                            q_tokens=q_token2,
                                                                            t=t, 
                                                                            device=device, 
                                                                            cache=cache)

    print("smpl pose shape:", smpl_pose.shape)    # B*(T-max_memt), 144
    print("smpl shape shape:", smpl_shape.shape)    # B*(T-max_memt), 10
    print("smpl cam shape:", smpl_cam.shape)    # B*(T-max_memt), 3 NOTE(yiwen) cut the first a couple of frames(max_cache) in GT
