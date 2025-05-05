import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cond):
        B, T, D = x.shape  # x: (B, T, D)
        B, T1, D = cond.shape  # cond: (B, T1, D)
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        K = self.k_proj(cond).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T1, head_dim)
        V = self.v_proj(cond).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T1, head_dim)

        attn_weights = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, T, T1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ V  # (B, n_heads, T, head_dim)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cond):
        x = x + self.dropout(self.self_attn(x, x, x)[0])  # Self-Attention
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attn(x, cond))  # Cross-Attention
        x = self.norm2(x)

        x = x + self.dropout(self.ffn(x))  # FFN
        x = self.norm3(x)
        return x

class ShortWindowTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return x


def main():
    batch_size = 64
    seq_len = 16    # Length of main sequence
    cond_len = 16    # Length of condition sequence
    feature_dim = 512
    d_model = 512
    n_heads = 2
    num_layers = 3

    # Create random test input
    x = torch.randn(batch_size, seq_len, feature_dim)  # (B, T, D)
    cond = torch.randn(batch_size, cond_len, feature_dim)  # (B, T1, D)

    print("Testing Transformer...")
    transformer = ShortWindowTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers)
    transformer_output = transformer(x, cond)
    print("Transformer Output Shape:", transformer_output.shape)  # Expected: (B, T, D)


if __name__ == "__main__":
    main()