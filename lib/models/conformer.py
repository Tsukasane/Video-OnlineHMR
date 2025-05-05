import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dim=512, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // n_heads
        
        self.q_proj = nn.Linear(d_model, attn_dim)
        self.k_proj = nn.Linear(d_model, attn_dim)
        self.v_proj = nn.Linear(d_model, attn_dim)
        self.out_proj = nn.Linear(attn_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cond):
        B, T, D = x.shape  # x: (B, T, D)
        B, T1, D = cond.shape  # cond: (B, T1, D)
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        K = self.k_proj(cond).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T1, head_dim)
        V = self.v_proj(cond).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T1, head_dim)

        attn_weights = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, T, T1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ V  # (B, n_heads, T, head_dim) ([64, 4, 2, 128])
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.attn_dim)  # (B, T, D)
        return self.out_proj(attn_output)
    
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model * 2, kernel_size=kernel_size, groups=d_model, padding=kernel_size//2) # TODO(yiwen) check this super large kernel size
        self.pointwise_conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # -->(B, D, T)
        x = self.pointwise_conv1(x) # [B, 2*D, T] kernel size=1, just to adjust the channel
        x = self.glu(x) # [B, D, T]

        x = self.depthwise_conv(x) # B, 2*D, T
        x = self.pointwise_conv2(x) # B, D, T
        x = x.transpose(1, 2)  # (B, T, D)
        return self.dropout(x) # ([2, 10, 64])

class ConformerBlock(nn.Module):
    '''
    Using a 1D Conv to aggregate information from the previous <time_windon> frames --> memory bank
    Then use Cross-attention to fuse info between memory band and (current, future frames)

    The conv+attn architecture is very similar to the conformer, but the Conv is not conducted on x after SA/CA, instead, Conv is for cond and conducted before CA
    '''
    def __init__(self, d_model, n_heads, kernel_size=31, attn_dim=512, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.conv = ConvModule(d_model, kernel_size, dropout) # using the conv part to build temporal dependency between different frames across the condition
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, n_heads, attn_dim, dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        x = x + self.dropout(self.ffn1(x))  # FFN
        x = self.norm1(x)

        ## To aggregate previous frames information
        # the output would be the same size as input
        cond = cond + self.dropout(self.conv(cond))  # Convolution Module
        cond = self.norm4(cond)
        # output a fix size cond as memory bank

        # NOTE(yiwen) if x is (current frame || future frame), shouldn't have non-causal format of self-attn (which is not supporting online)
        # x = x + self.dropout(self.self_attn(x, x, x)[0])  # Self-Attention
        # x = self.norm2(x)

        x = x + self.dropout(self.cross_attn(x, cond))  # Cross-Attention
        x = self.norm3(x) # [2, 10, 64]

        # x = x + self.dropout(self.conv(x))  # Convolution Module
        # x = self.norm4(x)

        x = x + self.dropout(self.ffn2(x))  # FFN
        return x

class Conformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers=6, kernel_size=3, attn_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, kernel_size, attn_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return x


def main():
    '''
    x:    B, T, D,  64, 2, 88
    cond: B, T1, D, 64, 3, 88
    '''
    batch_size = 64
    seq_len = 2    # current+future frames as target
    cond_len = 3    # previous cond_len frames as input
    feature_dim = 88    #
    d_model = 88    # input dim for FFN
    n_heads = 4
    num_layers = 3

    # Create random test input
    x = torch.randn(batch_size, seq_len, feature_dim)  # (B, T, D)
    cond = torch.randn(batch_size, cond_len, feature_dim)  # (B, T1, D)

    print("\nTesting Conformer...")
    conformer = Conformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers)
    conformer_output = conformer(x, cond)
    print("Conformer Output Shape:", conformer_output.shape)  # Expected: (B, T, D)


if __name__ == "__main__":
    main()