import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DCT(nn.Module):
    """
    Discrete Cosine Transform (DCT) and Inverse DCT (IDCT) implementation.
    Using the standard Type-II DCT.
    """
    def __init__(self):
        super(DCT, self).__init__()

    def forward(self, x):
        return self.dct_2d(x)

    def dct_1d(self, x, norm='ortho'):
        """
        1D DCT along the last dimension.
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= math.sqrt(N) * 2
            V[:, 1:] /= math.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def idct_1d(self, X, norm='ortho'):
        """
        1D IDCT along the last dimension.
        """
        x_shape = X.shape
        N = x_shape[-1]
        X_v = X.contiguous().view(-1, N) / 2

        if norm == 'ortho':
            X_v[:, 0] *= math.sqrt(N) * 2
            X_v[:, 1:] *= math.sqrt(N / 2) * 2

        k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=1)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def dct_2d(self, x, norm='ortho'):
        """
        2D DCT.
        """
        X1 = self.dct_1d(x, norm=norm)
        X2 = self.dct_1d(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm='ortho'):
        """
        2D IDCT.
        """
        x1 = self.idct_1d(X, norm=norm)
        x2 = self.idct_1d(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

class HFP(nn.Module):
    """
    High-Frequency Perception Module (HFP)
    """
    def __init__(self, in_channels=256, alpha=0.25, k=16):
        super(HFP, self).__init__()
        self.alpha = alpha
        self.k = k
        self.dct = DCT()
        
        # Channel Path (CP)
        # Groups=8 for 1x1 convolution
        self.cp_conv1_avg = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=8, bias=False)
        self.cp_conv1_max = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=8, bias=False)
        
        self.cp_conv2 = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # Spatial Path (SP)
        self.sp_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Feature Fusion
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        
        # Pooling for CP
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((k, k))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((k, k))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. High Frequency Response Generation ---
        # DCT Transform
        dct_feat = self.dct.dct_2d(x)
        
        # Generate High-Pass Filter
        # u < alpha * H, v < alpha * W -> 0, else 1
        mask = torch.ones((H, W), device=x.device, dtype=x.dtype)
        h_cutoff = int(self.alpha * H)
        w_cutoff = int(self.alpha * W)
        mask[:h_cutoff, :w_cutoff] = 0
        
        # Apply Filter
        filtered_dct = dct_feat * mask.unsqueeze(0).unsqueeze(0)
        
        # Inverse DCT
        high_freq_feat = self.dct.idct_2d(filtered_dct)
        
        # --- 2. Channel Path (CP) ---
        # Resize to k x k (16 x 16)
        feat_avg = self.adaptive_avg_pool(high_freq_feat) # B, C, k, k
        feat_max = self.adaptive_max_pool(high_freq_feat) # B, C, k, k
        
        # Sum over spatial dimensions after ReLU (Step 2 description is slightly ambiguous about ReLU timing, 
        # but "After ReLU activation, sum for each channel" implies ReLU first? 
        # Or does it mean "After pooling... -> ReLU -> Sum"? 
        # "After ReLU activation, sum for each channel..." usually implies ReLU(Features) -> Sum.
        # But features are likely signed after IDCT. 
        # Let's assume ReLU applied to the pooled features before summing.
        
        # Actually, looking at standard attention modules (like CBAM), usually it is MLP(Avg) + MLP(Max).
        # Text says: "After ReLU activation, sum for each channel, generating 2 1D vectors".
        # This implies: Pooling -> ReLU -> Sum -> Vector.
        
        feat_avg_act = self.relu(feat_avg)
        feat_max_act = self.relu(feat_max)
        
        vec_avg = feat_avg_act.sum(dim=(2, 3)) # B, C
        vec_max = feat_max_act.sum(dim=(2, 3)) # B, C
        
        # Reshape for 1D Conv: B, C, 1
        vec_avg = vec_avg.unsqueeze(2)
        vec_max = vec_max.unsqueeze(2)
        
        # Pass through 1x1 Group Conv
        score_avg = self.cp_conv1_avg(vec_avg)
        score_max = self.cp_conv1_max(vec_max)
        
        # Concat and final 1x1 Conv
        concat_score = torch.cat([score_avg, score_max], dim=1) # B, 2C, 1
        cp_weight = self.cp_conv2(concat_score) # B, C, 1
        
        # Broadcast and Multiply (C x 1 x 1 broadcast to C x H x W)
        cp_out = x * cp_weight.unsqueeze(3)
        
        # --- 3. Spatial Path (SP) ---
        # 1x1 Conv C->1
        sp_map = self.sp_conv(high_freq_feat) # B, 1, H, W
        sp_mask = self.sigmoid(sp_map)
        
        # Broadcast and Multiply
        sp_out = x * sp_mask
        
        # --- 4. Feature Fusion ---
        fusion_in = cp_out + sp_out
        out = self.fusion_conv(fusion_in)
        
        return out

class SDP(nn.Module):
    """
    Spatial Dependency Perception Module (SDP)
    """
    def __init__(self, in_channels=256, c5_size=(25, 20)):
        super(SDP, self).__init__()
        self.in_channels = in_channels
        self.c5_h, self.c5_w = c5_size
        
        # Q generation from Ci (Lower feature)
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # K, V generation from Pi+1 (Upper upsampled feature)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.scale = math.sqrt(in_channels)

    def forward(self, c_i, p_next_up):
        """
        c_i: Lower layer feature (Target) [B, C, H, W]
        p_next_up: Upper layer upsampled feature [B, C, H, W]
        """
        B, C, H, W = c_i.shape
        
        # --- 1. Q/K/V Generation ---
        Q = self.conv_q(c_i)       # B, C, H, W
        K = self.conv_k(p_next_up) # B, C, H, W
        V = self.conv_v(p_next_up) # B, C, H, W
        
        # --- 2. Feature Partitioning ---
        # Calculate number of blocks
        # n = (H / H5) * (W / W5)
        # We need to reshape to [B, num_blocks, C, block_h * block_w] or similar
        # To strictly follow "divided into n feature blocks... each block reshaped to 256 x (H5 x W5)"
        
        # Check if divisible
        if H % self.c5_h != 0 or W % self.c5_w != 0:
            # The paper says "Block size enforced to match C5... ensure divisible without residual".
            # In practice, we might need padding if it doesn't match, but the constraint says "Enforced".
            # We assume inputs satisfy this or we adapt. 
            # For this implementation, I will assume valid input or handle via Unfold.
            pass

        # Use unfold to get blocks? Or just view/permute.
        # Let's try view/permute which is faster if dimensions match.
        # H = n_h * h_block
        # W = n_w * w_block
        n_h = H // self.c5_h
        n_w = W // self.c5_w
        
        # Q: [B, C, n_h * h_block, n_w * w_block]
        # -> [B, C, n_h, h_block, n_w, w_block]
        # -> [B, n_h, n_w, C, h_block, w_block]
        # -> [B, n_h * n_w, C, h_block * w_block]
        
        def partition(feat):
            # feat: B, C, H, W
            feat = feat.view(B, C, n_h, self.c5_h, n_w, self.c5_w)
            feat = feat.permute(0, 2, 4, 1, 3, 5).contiguous() # B, n_h, n_w, C, h_block, w_block
            feat = feat.view(B, n_h * n_w, C, self.c5_h * self.c5_w) # B, N, C, L (L = H5*W5)
            return feat

        Q_blocks = partition(Q) # B, N, C, L
        K_blocks = partition(K) # B, N, C, L
        V_blocks = partition(V) # B, N, C, L
        
        # --- 3. Pixel-level Cross Attention ---
        # a_j = Softmax(q_j * k_j^T / sqrt(C))
        # q_j: C x L (from description "reshaped to 256 x (H5xW5)")
        # Actually for matrix multiplication usually we want (L x C) * (C x L) -> L x L attention map?
        # Or (C x L) * (L x C) -> C x C attention map?
        # "Pixel-level cross attention" implies attention between pixels in the block.
        # Block size L pixels.
        # Formula: q_j * k_j^T. q_j is 256 x L. k_j^T would be L x 256. Result 256x256. This is Channel attention.
        # Wait. "Pixel-level cross attention" usually means spatial attention.
        # Let's check the formula image.
        # Formula: a_j = Softmax( (q_j x k_j^T) / sqrt(256) )
        # If q_j is 256 x L, then q_j x k_j^T (where k_j is 256 x L, so k_j^T is L x 256) -> 256 x 256.
        # This looks like Channel Attention (covariance).
        # However, the title says "Pixel-level".
        # If it was spatial, it would be (L x 256) x (256 x L) -> L x L.
        # Let's look at "Output: Fusion spatial dependency feature block... Recover to full feature map".
        # If result is 256x256, multiplied by v_j (256xL), we get 256xL. This matches the shape.
        # So it is indeed calculating a correlation matrix of size C x C for each block?
        # Or is q_j actually L x 256?
        # Description: "Each block reshaped to 256 x (H5 x W5) (i.e., pixels in block flattened to vector)".
        # This phrasing "pixels... flattened to vector" usually means the vector dimension is the pixel dimension.
        # So shape is (C, L).
        # If shape is (C, L), then (C, L) x (L, C) -> (C, C).
        # Then (C, C) x (C, L) -> (C, L).
        # This is computationally cheaper if L > C. Here L = 25*20 = 500 > 256.
        # So it's likely (C, C) attention map.
        # BUT, "Pixel-level" name is confusing if it's channel attention.
        # Let's re-read carefully.
        # "Solving FPN upsampling misalignment... learn spatial dependency of lower feature Ci and upper feature Pi+1".
        # Usually misalignment is spatial.
        # Maybe the notation q_j * k_j^T implies the standard Attention(Q, K, V) = Softmax(QK^T)V.
        # In Transformers (ViT), Q is N x D. QK^T is N x N (Spatial x Spatial).
        # Here, if Q is C x L.
        # If we do Q^T K -> L x L (Spatial x Spatial).
        # If we do Q K^T -> C x C (Channel x Channel).
        # The formula explicitly says $q_j \times k_j^T$. With $q_j$ being $256 \times (H_5 \times W_5)$.
        # That is mathematically $(C \times L) \times (C \times L)^T = (C \times L) \times (L \times C) = C \times C$.
        # So it computes a channel-wise correlation matrix for each spatial block?
        # That seems to mix channel information based on spatial alignment?
        # Let's stick to the formula literally: $q_j \times k_j^T$ with dimensions given.
        # Result: C x C attention map.
        # Then multiply by $v_j$ (C x L).
        # $(C \times C) \times (C \times L) \rightarrow (C \times L)$.
        # Output shape matches input block shape.
        
        # B, N, C, L
        Q_blocks_matmul = Q_blocks # B, N, C, L
        K_blocks_matmul = K_blocks.transpose(2, 3) # B, N, L, C
        
        # Attention map: (B, N, C, L) @ (B, N, L, C) -> (B, N, C, C)
        attn = torch.matmul(Q_blocks_matmul, K_blocks_matmul)
        attn = attn / self.scale
        attn = F.softmax(attn, dim=-1) # Softmax along the last dimension (C)
        
        # Apply to V: (B, N, C, C) @ (B, N, C, L) -> (B, N, C, L)
        # Wait, if V is C x L. (C x C) @ (C x L) works.
        out_blocks = torch.matmul(attn, V_blocks)
        
        # --- 4. Feature Fusion Output ---
        # Recover to full feature map
        # B, N, C, L -> B, C, H, W
        def reconstruction(blocks):
            # blocks: B, N, C, L
            blocks = blocks.view(B, n_h, n_w, C, self.c5_h, self.c5_w)
            blocks = blocks.permute(0, 3, 1, 4, 2, 5).contiguous() # B, C, n_h, h_block, n_w, w_block
            feat = blocks.view(B, C, H, W)
            return feat

        sdp_out = reconstruction(out_blocks)
        
        # Add to original Ci
        out = sdp_out + c_i
        
        return out

