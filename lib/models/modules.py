import numpy as np
import einops
import torch
import torch.nn as nn
from .components.pose_transformer import TransformerDecoder
from .conformer import Conformer
from .transformer import ShortWindowTransformer
import matplotlib.pyplot as plt


class SMPLTransformerDecoderHead(nn.Module):
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
            context_dim = 1280,
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

        mean_params = np.load('data/smpl/smpl_mean_params.npz')
        init_body_pose = torch.tensor(mean_params['pose'], dtype=torch.float32).unsqueeze(0)
        init_betas = torch.tensor(mean_params['shape'], dtype=torch.float32).unsqueeze(0)
        init_cam = torch.tensor(mean_params['cam'], dtype=torch.float32).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose) # NOTE(yiwen) fix constant, flexibly switch device with model
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        
    def forward(self, x, **kwargs):
        '''
        Args:
            - B: the real batch size
            - N: window num
        '''

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c') # b=BN, HW, C TODO(yiwen) 把N分出来写seq的循环，或者把head直接一起写到st_module里

        init_body_pose = self.init_body_pose.expand(batch_size, -1) # NOTE(yiwen) originally use mean init, TODO switch to prev window init?
        init_betas = self.init_betas.expand(batch_size, -1) # they are stacked and passed to smplhead at the same time in training (so that cannot pass prev window motion to the next window)
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

def plot_attention_heads(attn_map, save_path=None):
    num_heads = attn_map.shape[0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i in range(num_heads):
        ax = axs[i // 2, i % 2]
        im = ax.imshow(attn_map[i].detach().cpu(), cmap="magma", aspect='auto')
        ax.set_title(f"Head {i}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


class temporal_attention_sw(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, is_img=False, head_dim=1):
        super(temporal_attention_sw, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.is_img = is_img
        self.l11 = nn.Linear(in_dim, hdim)
        self.l12 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        self.conformer = Conformer(d_model=512, n_heads=1, num_layers=3)
        self.naive_transfomer = ShortWindowTransformer(d_model=hdim, n_heads=4, num_layers=6) # 4

        self.frame_chunk_size = 1

        # img
        self.expanded_tem_idim = 3
        self.out_h = 16 # div 4, 2, 1
        self.out_w = 12
        self.compacted_spa_idim = 16
        self.tem_expansion_layer = nn.Linear(self.frame_chunk_size, self.expanded_tem_idim)
        self.tem_compact_layer = nn.Linear(self.expanded_tem_idim, 3*self.frame_chunk_size)
        self.spa_compact_layer1 = nn.Linear(192, self.compacted_spa_idim)
        self.spa_compact_layer2 = nn.Linear(192, self.compacted_spa_idim)

        self.spa_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(self.out_h, self.out_w))

        # self.spa_expansion_layer = nn.Linear(self.compacted_spa_idim, 192) #v3
        self.spa_expansion_layer = nn.Linear(self.out_h*self.out_w, 192) #v4

        # motion
        self.expanded_tem_mdim = 6 # NOTE(yiwen) tune para here12 15 18 24
        self.tem_expansion_layer1 = nn.Linear(self.frame_chunk_size, self.expanded_tem_mdim)
        self.tem_expansion_layer2 = nn.Linear(self.frame_chunk_size, self.expanded_tem_mdim)
        self.tem_compact_layer1 = nn.Linear(self.expanded_tem_mdim, 3*self.frame_chunk_size)
        self.tem_compact_layer2 = nn.Linear(self.expanded_tem_mdim, head_dim)

        self.feature_fusion = nn.Linear(2*hdim, hdim)
        self.pos_drop = nn.Dropout(0.15)

        TranLayer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu')
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)
        
        nn.init.xavier_uniform_(self.l11.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l12.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

        self.memory_num = 1 # NOTE(yiwen) fix memory length now

        weight = 1.0 / self.memory_num
        self.weight_list = [weight]

    def forward(self, x, mix_feats=None, is_train=False, is_valid=False, visualize_attention=False):
        '''
        Args:
            - [Image] x: (bnt) (hw) c  b=24 t=3
            - mix_feats: pass transformer feature across different forward terms in test
                         not in train / val times.
        Returns:
            - [Image] out: (bhw) t c-3
        '''
        if not self.is_img: # for pose, not used (setting motion_module=False in config)

            px = self.tem_expansion_layer1(x[:,0:1,:].permute(0,2,1)).permute(0,2,1) # prev_frame
            cx = self.tem_expansion_layer2(x[:,1:2,:].permute(0,2,1)).permute(0,2,1) # curr_frame
            del x

            ph = self.l11(px) # 4608, 16, 512
            ch = self.l12(cx)
            del px
            del cx
            
            # TODO(yiwen) check positional encoding after linear
            ph = self.pos_drop(ph)
            transformer_output = self.naive_transfomer(ch, ph)

            h = self.l2(transformer_output) # 4608, 16, 1280
            out = self.tem_compact_layer2(h.permute(0,2,1)).permute(0,2,1) # NOTE(yiwen) only the current

        else: # for img feature      
            B, _, hw, D = x.shape # B, 72, 192, 1283
            x = x.reshape(B, -1, 3, hw, D).transpose(0,1) # N, B, 3, hw, D
            memory = []
            out_ls = []

            for window_id in range(x.shape[0]):
                batch_windows = x[window_id]

                prev_frame = batch_windows[:,0:1,:,:].reshape(-1, hw, D)
                curr_frame = batch_windows[:,1:2,:,:].reshape(-1, hw, D)
                
                # hw 192 --> 12
                prev_frame = prev_frame.permute(0,2,1).reshape(B, D, 16, 12)
                prev_frame = self.spa_pooling_layer(prev_frame).reshape(B, D, -1).permute(0,2,1) # B, 12, D

                curr_frame = curr_frame.permute(0,2,1).reshape(B, D, 16, 12)
                curr_frame = self.spa_pooling_layer(curr_frame).reshape(B, D, -1).permute(0,2,1) # B, 12, D

                # feature compress
                ph = self.l11(prev_frame) # 16, 12, 512
                ch = self.l12(curr_frame)

                # TODO(yiwen) check positional encodding after linear
                ph = self.pos_drop(ph)

                if visualize_attention:
                    print(f"visualize_attention")
                    transformer_output, self_attns, cross_attns = self.naive_transfomer(ch, ph, return_attn=True)
                    # cross_attns[layer_idx]: shape [B, H, T_q, T_k]
                    crossattn_map = cross_attns[1][0]  # shape: [head, T_query, T_key]
                    selfattn_map = self_attns[0][:4]
                    plot_attention_heads(crossattn_map, "vis_crossattn.png")
                    plot_attention_heads(selfattn_map, "vis_selfattn.png")

                transformer_output = self.naive_transfomer(ch, ph) # NOTE(yiwen) ph.shape=ch.shape=transformer_output.shape=16, 12, 512
                inference_memory = transformer_output

                if (not is_train) and (not is_valid) and (mix_feats is not None):
                    memory.append(mix_feats)
                
                if len(memory)!=0:
                    mix_feats = sum(w * m for w, m in zip(self.weight_list, memory))
                    try:
                        transformer_output = torch.cat([transformer_output, mix_feats], dim=-1)
                        transformer_output = self.feature_fusion(transformer_output)
                    except:
                        import pdb
                        pdb.set_trace()

                else:  
                    # FIFO
                    memory.append(transformer_output)
                    if len(memory)>self.memory_num:
                        memory.pop(0)

                # NOTE(yiwen) currently, is bs=1 get the output image transformer, and bs=1 get the output posetransformer, then pass to the next window
                transformer_output = self.spa_expansion_layer(transformer_output.permute(0,2,1)).permute(0,2,1) # B, 192, 512
                h = self.l2(transformer_output) # B, 192, 1280

                curr_rp = h.reshape(-1, self.frame_chunk_size, h.shape[-1]) # 4608, 1, 1280
                out = self.tem_expansion_layer(curr_rp.permute(0,2,1)).permute(0,2,1) # bhw=384, t=3, c=1280

                # N, 384, 3, 1280
                out_ls.append(out)

            out_tensor = torch.stack(out_ls).to(out.device)
            N, _, _, D = out_tensor.shape
            out_tensor = out_tensor.reshape(N, B, hw, -1, D).transpose(0,1).reshape(-1, 3, D)

        return out_tensor, inference_memory


class temporal_attention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False):
        super(temporal_attention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu')
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)

        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def forward(self, x):
        x = x.permute(1,0,2)  # (b,t,c) -> (t,b,c)

        h = self.l1(x)
        h = self.pos_embedding(h)
        h = self.trans(h)
        h = self.l2(h)

        if self.residual:
            x = x[..., :self.out_dim] + h
        else:
            x = h
        x = x.permute(1,0,2)

        return x


class causal_attention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False, causal=False):
        super(causal_attention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.causal = causal
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(
            d_model=hdim, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, activation='gelu'
        )
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)
        
        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def generate_causal_mask(self, sz):
        # mask shape: (sz, sz), True means "block"
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, x):
        '''
        Args:
            - x: B, T, 147
        '''
        x = x.permute(1, 0, 2)  # (b, t, c) -> (t, b, c)

        h = self.l1(x)
        h = self.pos_embedding(h)

        if self.causal:
            seq_len = h.size(0)
            mask = self.generate_causal_mask(seq_len).to(h.device)
        else:
            mask = None

        h = self.trans(h, mask=mask)
        h = self.l2(h)

        if self.residual:
            x = x[..., :self.out_dim] + h
        else:
            x = h

        x = x.permute(1, 0, 2)  # back to (b, t, c)
        return x

 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
