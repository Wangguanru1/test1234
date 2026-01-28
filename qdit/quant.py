import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm
from quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda

class BalancedClustering(nn.Module):
    def __init__(self, n_channels, feature_dim, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

        # 聚类中心：可学习参数
        self.centers = nn.Parameter(torch.randn(n_clusters, feature_dim))

        # assignment logits: 可学习参数
        self.assignment_logits = nn.Parameter(torch.randn(n_channels, n_clusters))

    def forward(self, features):
        """
        features: (n_channels, feature_dim)
        """
        self.centers.data = self.centers.data.to(features.device)
        self.assignment_logits.data = self.assignment_logits.data.to(features.device)
        n, k = features.size(0), self.n_clusters

        # 计算距离矩阵 (n, k)
        dists = torch.cdist(features, self.centers, p=2) ** 2

        # softmax 分配概率 (n, k)
        assign_probs = F.softmax(self.assignment_logits, dim=1)

        # 主损失：加权距离
        loss_dist = torch.sum(assign_probs * dists)
        # one-hot 正则：熵最小化
        entropy = -torch.sum(assign_probs * torch.log(assign_probs + 1e-8), dim=1)
        loss_sharp = torch.mean(entropy)

        # 均衡性正则：每个中心接收通道数的方差
        per_cluster_weight = torch.sum(assign_probs, dim=0)  # (k,)
        loss_balance = torch.var(per_cluster_weight)

        return loss_dist, loss_sharp, loss_balance, assign_probs

def _compute_permutation_indices(assigned_cluster, features):
        """
        输入:
            assigned_cluster: shape = [n], 每个通道所属的聚类中心编号
            features: shape = [n, feature_dim], 每个通道的特征
        输出:
            perm_indices: shape = [n]，压缩的置换索引，表示每个位置对应的原始索引
            inverse_perm_indices: shape = [n]，逆置换索引
        """
        n = assigned_cluster.shape[0]
        n_clusters = int(assigned_cluster.max().item()) + 1
        
        # 计算每个聚类中各通道最大值与最小值之间距离的平均值
        cluster_ranges = []
        for cluster_id in range(n_clusters):
            cluster_mask = (assigned_cluster == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_features = features[cluster_mask]  # shape: [cluster_size, feature_dim]
                # 对于每个通道，计算其各特征维度的最大值和最小值之间的距离
                channel_ranges = []
                for channel_idx in range(cluster_features.shape[0]):
                    channel_feature = cluster_features[channel_idx]  # shape: [feature_dim]
                    channel_range = (channel_feature.max() - channel_feature.min()).item()
                    channel_ranges.append(channel_range)
                # 计算该聚类中所有通道距离的平均值
                cluster_avg_range = sum(channel_ranges) / len(channel_ranges) if channel_ranges else 0
                cluster_ranges.append((cluster_avg_range, cluster_id))
        
        # 按距离平均值从小到大排序聚类
        cluster_ranges.sort(key=lambda x: x[0])
        cluster_order = [cluster_id for _, cluster_id in cluster_ranges]
        
        # 重新分配聚类编号，使其按平均值顺序排列
        new_assigned_cluster = torch.zeros_like(assigned_cluster)
        for new_id, old_id in enumerate(cluster_order):
            new_assigned_cluster[assigned_cluster == old_id] = new_id
        
        # 获取新的通道排序索引：把同一聚类中心的通道排在一起
        perm_indices = torch.argsort(new_assigned_cluster)
        
        # 计算逆置换索引，确保在同一个设备上
        inverse_perm_indices = torch.zeros_like(perm_indices)
        inverse_perm_indices[perm_indices] = torch.arange(n, device=assigned_cluster.device, dtype=perm_indices.dtype)

        return perm_indices, inverse_perm_indices

def optimize_groups_assignment(features,groupsize):
    n_channels = features.shape[0]
    feature_dim = features.shape[1]
    n_clusters = features.shape[0] // groupsize
    model = BalancedClustering(n_channels=n_channels, feature_dim=feature_dim, n_clusters=n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    for step in range(600):
        loss_dist, loss_sharp, loss_balance, assign_probs = model(features)
        loss = loss_dist + 0.3 * loss_sharp + 1 * loss_balance

        optimizer.zero_grad()
        #print(f'loss:{loss}, loss_dist:{loss_dist}, loss_sharp:{loss_sharp}, loss_balance:{loss_balance}')
        loss.backward()
        optimizer.step()

    assign_probs = F.softmax(model.assignment_logits, dim=1)
    max_index = torch.argmax(assign_probs,dim=1)  
    perm_indices, inverse_perm_indices = _compute_permutation_indices(max_index, features)
    return perm_indices, inverse_perm_indices

def optimize_timestep_groups_assignment(features,groupsize):
    """
    为不同时间步优化分组分配，返回置换索引列表
    
    Returns:
        list: 包含不同时间步置换索引的列表，按时间步顺序排列
    """
    permutation_indices = []
    inverse_permutation_indices = []
    timesteps = sorted([int(t) for t in features.keys()])  # 确保按时间步顺序处理
    
    for timestep in tqdm(timesteps):
        
        timestep_features = features[timestep]  # shape: [n_channels, feature_dim]
        n_channels = timestep_features.shape[0]
        feature_dim = timestep_features.shape[1]
        n_clusters = max(1, n_channels // groupsize)  
        
        # 为当前时间步创建聚类模型
        model = BalancedClustering(n_channels=n_channels, feature_dim=feature_dim, n_clusters=n_clusters)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
        
        # 优化聚类分配
        for step in range(1000):
            loss_dist, loss_sharp, loss_balance, assign_probs = model(timestep_features)
            loss = loss_dist + 0.3 * loss_sharp + 1 * loss_balance

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 获取最终分配并计算置换索引
        assign_probs = F.softmax(model.assignment_logits, dim=1)
        max_index = torch.argmax(assign_probs, dim=1)  
        perm_indices, inverse_perm_indices = _compute_permutation_indices(max_index, timestep_features)
        permutation_indices.append(perm_indices)
        inverse_permutation_indices.append(inverse_perm_indices)
    return permutation_indices, inverse_permutation_indices


# Wrapper function for weight quantization
# Continous number of channel_group channels share the same quantization setup

def quantize_tensor_channel_group(W: torch.tensor, n_bits, group_size, tiling, sym, channel_group=1, clip_ratio=1.0, exponential=False, quant_type="int", quant_method="max") -> torch.tensor:
    assert W.is_contiguous(), "Input tensor is not contiguous"
    assert n_bits < 16

    if group_size > 0:
        assert W.shape[-1] % group_size == 0

    # group_size = 0 is per-channel quantization.
    if group_size == 0:
        W = quantize_tensor(W, n_bits=n_bits, group_size=0, tiling=tiling, sym=sym, exponential=exponential,quant_method=quant_method)
    else:
        for i1 in range(0, W.shape[1], group_size):
            i2 = min(i1 + group_size, W.shape[1])
            w = W[:,i1:i2]

            # Continous channels share the same quantization setup.
            # This trick is used for efficiency consideration.
            if channel_group > 1:
                w = w.reshape(int(W.shape[0]/channel_group), -1).contiguous() # Continous for bitsandbytes kernel calling

            # group_size is set to 0 because the layout is
            # already [num_groups, group_size]
            w = quantize_tensor(
                w,
                n_bits=n_bits,
                group_size=0,
                tiling=tiling,
                sym=sym,
                clip_ratio=clip_ratio,
                exponential=exponential,
                quant_type=quant_type,
                quant_method=quant_method
            )

            # Reshape back to original shape.
            if channel_group > 1:
                w = w.reshape(-1, group_size)
            W[:,i1:i2] = w

    return W.contiguous()



def quantize_tensor(w: torch.tensor, n_bits, group_size, tiling, sym, clip_ratio=1.0, exponential=False, quant_type="int", quant_method="max") -> torch.tensor:
    savedShape = w.shape
    w = w.squeeze()
    if not w.is_contiguous():
        w = w.contiguous()
    assert w.is_contiguous(), "tensor should be continous for bitsandbytes kernel."

    if tiling > 0:
        assert False, "16x16 Block-wise Quantization is abandoned in this project."

    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format should be: [num_groups, group_size]"
    assert n_bits < 16

    def lp_loss(pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        y = torch.flatten(x, 1)
        return y.mean(1,keepdim=True)
    
    
    assert quant_type == "int", "Options should be in [int, fp]"
    if quant_method == "max":
        if sym:
            w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        else:
            w_max = w.amax(dim=-1, keepdim=True)
            w_min = w.amin(dim=-1, keepdim=True)

        if sym:
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            if clip_ratio < 1.0:
                w_max = w_max * clip_ratio
            scales = w_max / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            if clip_ratio < 1.0:
                w_max *= clip_ratio
                w_min *= clip_ratio
            scales = (w_max-w_min).clamp(min=1e-5) / q_max
            base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    elif quant_method == "mse":
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        w_absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_score = torch.zeros_like(w_max) + (1e10)
        best_min = w_min.clone()
        best_max = w_max.clone()
        best_absmax = w_absmax.clone()
        for i in range(100):
            if sym:
                new_max = w_absmax * (1.0 - (i * 0.001))
                q_max = (2**(n_bits-1)-1)
                q_min = (-2**(n_bits-1))
                scales = new_max / q_max
                base = torch.zeros_like(scales)
            else:
                new_max = w_max * (1.0 - (i * 0.001))
                new_min = w_min * (1.0 - (i * 0.001))
                q_max = (2**(n_bits)-1)
                q_min = (0)
                scales = (new_max-new_min).clamp(min=1e-5) / q_max
                base = torch.round(-new_min/scales).clamp_(min=q_min, max=q_max)
            w_q = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(w, w_q, p=2.4)
            if sym:
                best_absmax = torch.where(score < best_score, new_max, best_absmax)
            else:
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(best_score, score)
        # print('clip_ratio:', (best_absmax/w_absmax)) 
        if sym: 
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            scales = best_absmax / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            scales = (best_max-best_min).clamp(min=1e-5) / q_max
            base = torch.round(-best_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales

    else:
        raise NotImplementedError
    
    return w.reshape(savedShape)

# Wrapper function for activation quantization
# Simulate mixed-precision by decomposing input

def quantize_activation_wrapper(x: torch.tensor, args) -> torch.tensor:
    if args.abits >= 16:
        return x 
    
    qFunction = partial(
        quantize_tensor, 
        n_bits=args.abits, 
        group_size=args.act_group_size, 
        tiling=args.tiling, 
        sym=args.a_sym,
        clip_ratio=args.a_clip_ratio,
        exponential=False,
        quant_type=args.quant_type
    )

    savedShape = x.shape
    x = x.reshape(-1, savedShape[-1])
    assert args.act_group_size == 0 or (savedShape[-1]) % args.act_group_size == 0
    
    x = qFunction(x)

    return x.view(savedShape)


def quantize_activation_per_channel_group(x: torch.tensor, args) -> torch.tensor:
    """
    对输入张量进行 per-channel-group 量化。
    在一个通道组上的所有 token 共享同一组量化参数。
    输入 x 的形状应为 [B, N, C]。
    """
    if args.abits >= 16:
        return x

    saved_shape = x.shape
    B, N, C = saved_shape
    group_size = args.act_group_size

    assert group_size > 0 and C % group_size == 0, \
        "act_group_size 必须大于0且能被C维度整除"
    
    # 将输入张量从 [B, N, C] 转换为 [C // group_size, B * N * group_size]
    # 这样每个通道组的所有数据都在一行中
    x_reshaped = x.view(B, N, C // group_size, group_size).permute(2, 0, 1, 3).contiguous()
    x_reshaped = x_reshaped.view(C // group_size, B * N * group_size)

    # 使用 quantize_tensor 进行量化，group_size 为 B * N * group_size，即整个通道组
    q_function = partial(
        quantize_tensor,
        n_bits=args.abits,
        group_size=0,  # 每一行是一个独立的量化组
        tiling=args.tiling,
        sym=args.a_sym,
        clip_ratio=args.a_clip_ratio,
        exponential=False,
        quant_type=args.quant_type,
        quant_method="max"  # 或者其他量化方法
    )

    quantized_x = q_function(x_reshaped)

    # 将量化后的张量恢复到原始形状
    quantized_x = quantized_x.view(C // group_size, B, N, group_size).permute(1, 2, 0, 3).contiguous()
    return quantized_x.view(saved_shape)


def quantize_activation_token_wise(x: torch.tensor, args, token_N=None) -> torch.tensor:
    """
    针对[B, N, C]的输入进行token-wise分组量化。
    将每个 [token_N, group_size] 的块被视为一个独立的量化组。
    """
    if args.abits >= 16:
        return x

    savedShape = x.shape
    B, N, C = savedShape
    group_size = args.token_group_size

    if token_N is None:
        token_N = N
    
    assert group_size > 0 and C % group_size == 0, \
        "token_group_size 必须大于0且能被C维度整除"
    assert N % token_N == 0, f"序列长度 N ({N}) 必须能被 token_N ({token_N}) 整除"

    # qFunction的group_size参数需要是每个组内元素的总数
    qFunction = partial(
        quantize_tensor,
        n_bits=args.abits,
        group_size=0,  # 每个组的大小是 token_N * group_size
        tiling=args.tiling,
        sym=args.a_sym,
        clip_ratio=args.a_clip_ratio,
        exponential=False,
        quant_type=args.quant_type,
        quant_method="max"
    )

    # Reshape to [B, N/token_N, token_N, C/group_size, group_size]
    x = x.view(B, N // token_N, token_N, C // group_size, group_size)
    # Permute to [B, N/token_N, C/group_size, token_N, group_size]
    x = x.permute(0, 1, 3, 2, 4)
    # Reshape to [B * (N/token_N) * (C/group_size), token_N * group_size]
    # 每一行是一个完整的 [token_N, group_size] 块
    x = x.reshape(B * (N // token_N) * (C // group_size), token_N * group_size)
    
    x = qFunction(x)

    # Reshape back to original shape
    x = x.view(B, N // token_N, C // group_size, token_N, group_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(savedShape)


def quantize_attn_v_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Input shape: [bsz, self.num_heads, seq_len, self.head_dim]
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)


def quantize_attn_k_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)


def quantize_attn_q_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

class Quantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x


    def forward(self, hidden_states):
        if self.args.static == False or self.scales is None:
            return self.act_quant(hidden_states)
            #return quantize_activation_token_wise(hidden_states, self.args)
            #return quantize_activation_per_channel_group(hidden_states, self.args)
            #return hidden_states
            

        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == False

        hidden_states = hidden_states.view(-1, savedShape[-1])
        selected_states = hidden_states.clone()

        if self.args.act_group_size > 0:
            selected_states = selected_states.reshape(-1, self.args.act_group_size)

        B, N, C = savedShape
        if self.args.act_group_size > 0:
            scales, base = self.scales[0].repeat(B * N, 1), self.scales[1].repeat(B * N, 1)
        else:
            scales, base = self.scales[0].unsqueeze(0).repeat(B * N, 1), self.scales[1].unsqueeze(0).repeat(B * N, 1)
        assert scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        selected_states = (torch.clamp(torch.round(selected_states / scales) + base, self.q_min, self.q_max) - base) * scales
        selected_states = selected_states.reshape(-1, savedShape[-1])
        hidden_states = selected_states
        
        return hidden_states.view(savedShape)
    
    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def extra_repr(self):
        if self.args.static == True:
            return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, scale={self.scales[0]}, base={self.scales[1]}'
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'
    
class RotateQuantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x 
        group_size = self.args.act_group_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.h1 = random_hadamard_matrix(group_size * 3, device)

    def forward(self, hidden_states):
        #print(hidden_states.dtype)
        # Hadamard旋转+量化+逆旋转
        if not hasattr(self, 'rotation_matrix') or self.rotation_matrix is None:
            self.rotation_matrix = random_hadamard_matrix(hidden_states.shape[-1], hidden_states.device)
        # 保证dtype一致
        rot_mat = self.rotation_matrix.to(hidden_states.dtype)
        # 1. Hadamard旋转
        rotated = torch.matmul(hidden_states, rot_mat)

        quantized = self.act_quant(rotated)
        # 3. Hadamard逆旋转
        inv_rot = rot_mat.t()
        de_rotated = torch.matmul(quantized, inv_rot)
        return de_rotated

        # savedShape = hidden_states.shape
        # if self.args.act_group_size > 0:
        #     total_channels = savedShape[-1]
        #     num_groups = total_channels // self.args.act_group_size
        # last_three_group_start = (num_groups - 3) * self.args.act_group_size
        # h1 = self.h1.to(hidden_states.dtype)
        # part = hidden_states[..., last_three_group_start:].clone()  # shape [..., 3*group_size]
        # transformed_part = torch.matmul(part, h1)
        # hidden_states[..., last_three_group_start:] = transformed_part

        # hidden_states = self.act_quant(hidden_states)

        # quantized_part = hidden_states[..., last_three_group_start:]
        # inversed_part = torch.matmul(quantized_part, h1.t())
        # hidden_states[..., last_three_group_start:] = inversed_part

        # return hidden_states

    
    def to(self, *args, **kwargs):
        super(RotateQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def extra_repr(self):
        if self.args.static == True:
            return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, scale={self.scales[0]}, base={self.scales[1]}'
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'
  

class PermutationQuantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.register_buffer('permutation_indices', None)
        self.register_buffer('inverse_permutation_indices', None)
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x


    def forward(self, hidden_states):
        #print(self.act_quant(hidden_states))
        if self.permutation_indices is not None:
            # 使用索引进行置换：hidden_states[..., permutation_indices]
            hidden_states = hidden_states[..., self.permutation_indices]
        if self.args.static == False or self.scales is None:
            #hidden_states = self.act_quant(hidden_states)

            savedShape = hidden_states.shape
            total_channels = savedShape[-1]
            num_groups = total_channels // self.args.act_group_size
            if num_groups > 2:
                # 取最后两组
                last_two_group_start = (num_groups - 3) * self.args.act_group_size
                last_two_groups = hidden_states[..., last_two_group_start:].clone()  # shape [..., 2*group_size]
                
                # 整体量化
                quantized = self.act_quant(hidden_states)
                
                
                quantized[..., last_two_group_start:] = last_two_groups
                hidden_states = quantized

            
            if self.inverse_permutation_indices is not None:
                # 使用逆置换索引恢复原始顺序
                hidden_states = hidden_states[..., self.inverse_permutation_indices]
                #print(hidden_states)
                return hidden_states
            return hidden_states
        
        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == False

        hidden_states = hidden_states.view(-1, savedShape[-1])
        selected_states = hidden_states.clone()

        if self.args.act_group_size > 0:
            selected_states = selected_states.reshape(-1, self.args.act_group_size)

        B, N, C = savedShape
        if self.args.act_group_size > 0:
            scales, base = self.scales[0].repeat(B * N, 1), self.scales[1].repeat(B * N, 1)
        else:
            scales, base = self.scales[0].unsqueeze(0).repeat(B * N, 1), self.scales[1].unsqueeze(0).repeat(B * N, 1)
        assert scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        selected_states = (torch.clamp(torch.round(selected_states / scales) + base, self.q_min, self.q_max) - base) * scales
        selected_states = selected_states.reshape(-1, savedShape[-1])
        hidden_states = selected_states
        hidden_states = hidden_states.view(savedShape)
        if self.inverse_permutation_indices is not None:
            # 使用逆置换索引恢复原始顺序
            hidden_states = hidden_states[..., self.inverse_permutation_indices]
        return hidden_states
    
    def to(self, *args, **kwargs):
        super(PermutationQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    

    def get_permutation_matrix(self,features):
        perm_indices, inverse_perm_indices = optimize_groups_assignment(features=features,groupsize=self.args.act_group_size)
        self.register_buffer('permutation_indices', perm_indices)
        self.register_buffer('inverse_permutation_indices', inverse_perm_indices)

    def extra_repr(self):
        if self.args.static == True:
            return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, scale={self.scales[0]}, base={self.scales[1]}'
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'
    
def rotate_and_quantize_groups(hidden_states, args, num_rotate_groups, rotation_matrix):
    """
    对指定数量的组进行旋转、量化和逆旋转。
    """
    group_size = args.act_group_size

    # 适应fc2
    if hidden_states.shape[-1] == 4608:
        group_size = args.act_group_size * 4
    
    num_groups = hidden_states.shape[-1] // group_size

    if args.fix_group_num:
        num_rotate_groups = 3
    

    last_groups_start = (num_groups - num_rotate_groups) * group_size
    
    # 旋转
    for i in range(num_rotate_groups):
        start_idx = last_groups_start + i * group_size
        end_idx = start_idx + group_size
        part = hidden_states[..., start_idx:end_idx]
        transformed_part = torch.matmul(part, rotation_matrix)
        hidden_states[..., start_idx:end_idx] = transformed_part

    # 量化
    #args.a_sym = True  # 对旋转后的部分使用对称量化
    #hidden_states = quantize_activation_wrapper(hidden_states, args)

    hidden_states[..., :last_groups_start] = quantize_activation_wrapper(hidden_states[..., :last_groups_start], args)
    args.a_sym = True  # 对旋转后的部分使用对称量化
    hidden_states[..., last_groups_start:] = quantize_activation_wrapper(hidden_states[..., last_groups_start:], args)
    args.a_sym = False  # 恢复原始设置
    # 逆旋转
    inv_rotation_matrix = rotation_matrix.t()
    for i in range(num_rotate_groups):
        start_idx = last_groups_start + i * group_size
        end_idx = start_idx + group_size
        quantized_part = hidden_states[..., start_idx:end_idx]
        inversed_part = torch.matmul(quantized_part, inv_rotation_matrix)
        hidden_states[..., start_idx:end_idx] = inversed_part
        
    return hidden_states


class TimestepPermutationQuantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.register_buffer('permutation_indices', None)  # shape: [num_timesteps, n_channels]
        self.register_buffer('inverse_permutation_indices', None)  # shape: [num_timesteps, n_channels]
        self.num_rotate_groups = 3
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x
        self.timestep = self.args.num_sampling_steps - 1
        #self.timestep = 0
        self.rotation_matrix = None
        self.save_act = False
        self.cali_gptq = False
        # 新增：用于保存每个时间步的激活值
        self._saved_activations = {}
        self.quant_mode = True
        self.group_size = self.args.act_group_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.h1 = random_hadamard_matrix(self.group_size, device)
        
        self.h2 = random_hadamard_matrix(self.group_size * 4, device)

        #print('hadamard matrix shape:', self.h1.shape)

        if self.args.not_compress:
            self.permutation_groups_size = 5
        elif self.args.num_sampling_steps == 50:
            self.permutation_groups_size = 5
        elif self.args.num_sampling_steps == 100:
            self.permutation_groups_size = 10

    def _save_activation(self, orig, permuted, timestep_idx):
        # 只保存每个时间步的第一个样本
        if timestep_idx not in self._saved_activations:
            # orig, permuted: [B, N, C] 或更高维，取第一个样本
            if orig.dim() > 2:
                orig_sample = orig[0].detach().cpu()
                perm_sample = permuted[0].detach().cpu()
            else:
                orig_sample = orig.detach().cpu()
                perm_sample = permuted.detach().cpu()
            self._saved_activations[timestep_idx] = {
                'timestep': timestep_idx,
                'original': orig_sample,
                'permuted': perm_sample
            }
            # 保存为pt文件
            torch.save(self._saved_activations[timestep_idx], f"/data1/clinic_rag/Q-DiT/activation/timestep_activation_{timestep_idx}.pt")

    def forward(self, hidden_states):
        if not self.quant_mode:
            return hidden_states
        if self.cali_gptq:
            return hidden_states
            #return quantize_activation_token_wise(hidden_states, self.args)

        orig_hidden = hidden_states.clone() if self.save_act else None
        # permutation transform
        hidden_states = hidden_states[..., self.permutation_indices[((self.timestep + self.permutation_groups_size) // self.permutation_groups_size) * self.permutation_groups_size - 1].long()]
        #hidden_states = hidden_states[..., self.permutation_indices[self.timestep // self.permutation_groups_size].long()]
        # fix timestep
        #hidden_states = hidden_states[..., self.permutation_indices[50].long()]
        # 保存原始和置换后的激活值
        if self.save_act:
            self._save_activation(orig_hidden, hidden_states, self.timestep)
        if self.args.static == False or self.scales is None:

            savedShape = hidden_states.shape
            if self.group_size > 0:
                total_channels = savedShape[-1]
                num_groups = total_channels // self.group_size
                if num_groups > 3:
                    # 取最后三组
                    if self.args.use_diagonal_block_matrix:
                        
                        if hidden_states.shape[-1] == 4608:
                            h2 = self.h2.to(hidden_states.dtype)
                            hidden_states = rotate_and_quantize_groups(hidden_states, self.args, num_rotate_groups=3, rotation_matrix=h2)
                        else:
                            h1 = self.h1.to(hidden_states.dtype)
                            hidden_states = rotate_and_quantize_groups(hidden_states, self.args, num_rotate_groups=self.num_rotate_groups, rotation_matrix=h1)
                    else:
                        # h1 = self.h1.to(hidden_states.dtype)
                        # part = hidden_states[..., last_three_group_start:].clone()  # shape [..., 3*group_size]
                        # transformed_part = torch.matmul(part, h1)
                        # hidden_states[..., last_three_group_start:] = transformed_part


                        #hidden_states = self.act_quant(hidden_states)
                        #hidden_states = quantize_activation_token_wise(hidden_states, self.args)

                        args = self.args
                        args.act_group_size = 128
                        hidden_states = quantize_activation_wrapper(hidden_states, args)
                        #hidden_states[..., :last_three_group_start] = quantize_activation_wrapper(hidden_states[..., :last_three_group_start], self.args)

                        # quantized_part = hidden_states[..., last_three_group_start:]
                        # inversed_part = torch.matmul(quantized_part, h1.t())
                        # hidden_states[..., last_three_group_start:] = inversed_part
            else:
                # 如果没有分组，进行原始的量化逻辑
                hidden_states = self.act_quant(hidden_states)

            #inverse permutation transform
            hidden_states = hidden_states[..., self.inverse_permutation_indices[((self.timestep + self.permutation_groups_size) // self.permutation_groups_size) * self.permutation_groups_size - 1].long()]
            #hidden_states = hidden_states[..., self.inverse_permutation_indices[self.timestep // self.permutation_groups_size].long()]
            # fix timestep
            #hidden_states = hidden_states[..., self.inverse_permutation_indices[50].long()]
            self.timestep -= 1
            if self.timestep < 0:
                #self.timestep = self.permutation_indices.shape[0] * self.permutation_groups_size - 1
                self.timestep = 49
            # self.timestep += 1
            # if self.timestep >= self.permutation_indices.shape[0] * self.permutation_groups_size:
            #     self.timestep = 0
            return hidden_states
        
        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == False

        # hidden_states = hidden_states.view(-1, savedShape[-1])
        # selected_states = hidden_states.clone()

        # if self.args.act_group_size > 0:
        #     selected_states = selected_states.reshape(-1, self.args.act_group_size)

        # B, N, C = savedShape
        # if self.args.act_group_size > 0:
        #     scales, base = self.scales[0].repeat(B * N, 1), self.scales[1].repeat(B * N, 1)
        # else:
        #     scales, base = self.scales[0].unsqueeze(0).repeat(B * N, 1), self.scales[1].unsqueeze(0).repeat(B * N, 1)
        # assert scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        # selected_states = (torch.clamp(torch.round(selected_states / scales) + base, self.q_min, self.q_max) - base) * scales
        # selected_states = selected_states.reshape(-1, savedShape[-1])
        # hidden_states = selected_states
        # hidden_states = hidden_states.view(savedShape)
        # if self.inverse_permutation_indices is not None:
        #     timestep_idx = self.timestep % self.inverse_permutation_indices.shape[0]
        #     # 使用逆置换索引恢复原始顺序
        #     hidden_states = hidden_states[..., self.inverse_permutation_indices[timestep_idx]]
        # self.timestep += 1
        
        # return hidden_states
    
    def to(self, *args, **kwargs):
        super(TimestepPermutationQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    

    def get_permutation_matrix(self, features):
        permutation_indices, inverse_permutation_indices = optimize_timestep_groups_assignment(features=features, groupsize=self.args.act_group_size)
        # 将列表转换为tensor并存储
        perm_tensor = torch.stack(permutation_indices, dim=0)
        inverse_perm_tensor = torch.stack(inverse_permutation_indices, dim=0)
        self.register_buffer('permutation_indices', perm_tensor)
        self.register_buffer('inverse_permutation_indices', inverse_perm_tensor)

    def extra_repr(self):
        if self.args.static == True:
            return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, scale={self.scales[0]}, base={self.scales[1]}'
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}, num_rotate_groups={self.num_rotate_groups}'
    
class PartialPrecisionQuantizer(nn.Module):
    """
    在指定时间步区间保持全精度，其他区间用普通量化。
    区间通过 args.precision_timestep_range 指定，如 [start, end]。
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        self.act_quant = lambda x: x
        # 区间参数，格式：[start, end]，如 [10, 20]
        self.precision_range = getattr(args, 'precision_timestep_range', [0, -1])
        self.timestep = 0

    def forward(self, hidden_states):
        # 判断当前时间步是否在全精度区间
        if self.precision_range[0] <= self.timestep < self.precision_range[1]:
            out = hidden_states
        else:
            out = self.act_quant(hidden_states)
        self.timestep += 1
        return out

    def to(self, *args, **kwargs):
        super(PartialPrecisionQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def extra_repr(self):
        return f'partial_precision_range={self.precision_range}, abits={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'


class ProgressiveTimestepPermutationQuantizer(nn.Module):
    """
    渐进式时间步置换量化器：
    - 前20%时间步：使用 act_group_size * 2 的组大小
    - 中间50%时间步：使用 act_group_size 的组大小  
    - 最后30%时间步：使用 act_group_size / 2 的组大小
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.register_buffer('permutation_indices', None)  # 2倍组大小的置换索引
        self.register_buffer('inverse_permutation_indices', None)
        
        self.args = args
        self.act_quant = lambda x: x
        self.timestep = 0
        self.total_timesteps = getattr(args, 'num-fid-samples', 100)  # 默认50个时间步
        
        self.save_act = False
        self._saved_activations = {}

    def _update_abits(self):
        
        if self.timestep == 10:
            self.args.abits = self.args.abits + 2
            self.configure(partial(quantize_activation_wrapper, args=self.args),
            None
            )
            
        if self.timestep == 80:
            self.args.abits = self.args.abits + 2
            self.configure(partial(quantize_activation_wrapper, args=self.args),
            None
            )
            
        self.timestep += 1

    def _save_activation(self, orig, permuted, timestep_idx):
        """保存激活值用于分析"""
        if timestep_idx not in self._saved_activations:
            if orig.dim() > 2:
                orig_sample = orig[0].detach().cpu()
                perm_sample = permuted[0].detach().cpu()
            else:
                orig_sample = orig.detach().cpu()
                perm_sample = permuted.detach().cpu()
            self._saved_activations[timestep_idx] = {
                'timestep': timestep_idx,
                'original': orig_sample,
                'permuted': perm_sample
            }
            torch.save(self._saved_activations[timestep_idx], f"/data1/clinic_rag/Q-DiT/activation/progressive_timestep_activation_{timestep_idx}.pt")

    def forward(self, hidden_states):
        # 获取当前时间步对应的组大小和类型
        self._update_abits()
        
        return self.act_quant(hidden_states)
        # orig_hidden = hidden_states.clone()
        
        # if self.perm_indices is not None:
        #     timestep_idx = self.timestep % self.perm_indices.shape[0]
        #     # 使用索引进行置换
        #     hidden_states = hidden_states[..., self.perm_indices[(timestep_idx // 10) * 10]]
        #     # 保存原始和置换后的激活值
        #     if self.save_act:
        #         self._save_activation(orig_hidden, hidden_states, timestep_idx)

        # if self.args.static == False or self.scales is None:
        #     savedShape = hidden_states.shape
        #     if self.args.act_group_size > 0:
        #         total_channels = savedShape[-1]
        #         num_groups = total_channels // self.args.act_group_size
        #         if num_groups > 3:
        #             # 取最后两组
        #             last_two_group_start = (num_groups - 3) * self.args.act_group_size
        #             last_two_groups = hidden_states[..., last_two_group_start:].clone()  # shape [..., 2*group_size]
        #             # Hadamard变换
        #             hadamard = self.rotation_matrix.to(last_two_groups.dtype)
        #             last_two_groups_h = torch.matmul(last_two_groups, hadamard)

        #             # 替换原始hidden_states的最后两组
        #             hidden_states_h = hidden_states.clone()
        #             hidden_states_h[..., last_two_group_start:] = last_two_groups_h
        #             # 整体量化
        #             quantized = self.act_quant(hidden_states_h)

        #             # 量化后对最后两组做Hadamard逆变换                   
        #             last_two_groups_q = quantized[..., last_two_group_start:]
        #             hadamard_inv = hadamard.t().to(last_two_groups_q.dtype)
        #             last_two_groups_q_inv = torch.matmul(last_two_groups_q, hadamard_inv)
        #             # 替换回去
        #             quantized[..., last_two_group_start:] = last_two_groups_q_inv
                    
        #             hidden_states = quantized

        #         else:
        #             # 如果组数不大于2，直接量化
        #             hidden_states = self.act_quant(hidden_states)
        #     else:
        #         # 如果没有分组，进行原始的量化逻辑
        #         hidden_states = self.act_quant(hidden_states)

        #     if self.inverse_permutation_indices is not None:
        #         timestep_idx = self.timestep % self.inverse_permutation_indices.shape[0]
        #         # 使用逆置换索引恢复原始顺序
        #         hidden_states = hidden_states[..., self.inverse_permutation_indices[(timestep_idx // 10) * 10]]
        #     self.timestep += 1
        #     return hidden_states
        
        
        # # Static quantization path (similar to original implementation)
        # savedShape = hidden_states.shape
        # assert self.scales is not None, "Scales is None"
        # assert self.args.a_sym == False

        
        

        # if self.args.static == False or self.scales is None:

        #     savedShape = hidden_states.shape
        #     if self.args.act_group_size > 0:
        #         total_channels = savedShape[-1]
        #         num_groups = total_channels // self.args.act_group_size
        #         if num_groups > 3:
        #             # 取最后两组
        #             last_two_group_start = (num_groups - 3) * self.args.act_group_size
        #             last_two_groups = hidden_states[..., last_two_group_start:].clone()  # shape [..., 2*group_size]
        #             # Hadamard变换
        #             hadamard = self.rotation_matrix.to(last_two_groups.dtype)
        #             last_two_groups_h = torch.matmul(last_two_groups, hadamard)

        #             # 替换原始hidden_states的最后两组
        #             hidden_states_h = hidden_states.clone()
        #             hidden_states_h[..., last_two_group_start:] = last_two_groups_h
        #             # 整体量化
        #             quantized = self.act_quant(hidden_states_h)

        #             # 量化后对最后两组做Hadamard逆变换                   
        #             last_two_groups_q = quantized[..., last_two_group_start:]
        #             hadamard_inv = hadamard.t().to(last_two_groups_q.dtype)
        #             last_two_groups_q_inv = torch.matmul(last_two_groups_q, hadamard_inv)
        #             # 替换回去
        #             quantized[..., last_two_group_start:] = last_two_groups_q_inv
                    
        #             hidden_states = quantized

                    
        #         else:
        #             # 如果组数不大于2，直接量化
        #             hidden_states = self.act_quant(hidden_states)
        #     else:
        #         # 如果没有分组，进行原始的量化逻辑
        #         hidden_states = self.act_quant(hidden_states)

        #     if self.inverse_permutation_indices is not None:
        #         timestep_idx = self.timestep % self.inverse_permutation_indices.shape[0]
        #         # 使用逆置换索引恢复原始顺序
        #         hidden_states = hidden_states[..., self.inverse_permutation_indices[(timestep_idx // 10) * 10]]
        #     self.timestep += 1
        #     return hidden_states
    
    def to(self, *args, **kwargs):
        super(ProgressiveTimestepPermutationQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def get_permutation_matrix(self, features):
        """
        为不同组大小生成置换矩阵
        features: 包含不同时间步特征的字典
        """
        # 为2倍组大小生成置换矩阵
        group_size_2x = self.args.act_group_size * 2
        permutation_indices_2x, inverse_permutation_indices_2x = optimize_timestep_groups_assignment(
            features=features, groupsize=group_size_2x)
        perm_tensor_2x = torch.stack(permutation_indices_2x, dim=0)
        inverse_perm_tensor_2x = torch.stack(inverse_permutation_indices_2x, dim=0)
        self.register_buffer('permutation_indices_2x', perm_tensor_2x)
        self.register_buffer('inverse_permutation_indices_2x', inverse_perm_tensor_2x)

        # 为标准组大小生成置换矩阵
        group_size_1x = self.args.act_group_size
        permutation_indices_1x, inverse_permutation_indices_1x = optimize_timestep_groups_assignment(
            features=features, groupsize=group_size_1x)
        perm_tensor_1x = torch.stack(permutation_indices_1x, dim=0)
        inverse_perm_tensor_1x = torch.stack(inverse_permutation_indices_1x, dim=0)
        self.register_buffer('permutation_indices_1x', perm_tensor_1x)
        self.register_buffer('inverse_permutation_indices_1x', inverse_perm_tensor_1x)

        # 为半组大小生成置换矩阵
        group_size_half = max(1, self.args.act_group_size // 2)
        permutation_indices_half, inverse_permutation_indices_half = optimize_timestep_groups_assignment(
            features=features, groupsize=group_size_half)
        perm_tensor_half = torch.stack(permutation_indices_half, dim=0)
        inverse_perm_tensor_half = torch.stack(inverse_permutation_indices_half, dim=0)
        self.register_buffer('permutation_indices_half', perm_tensor_half)
        self.register_buffer('inverse_permutation_indices_half', inverse_perm_tensor_half)

    def extra_repr(self):
        return f'progressive_groupsize=[{self.args.act_group_size*2}, {self.args.act_group_size}, {max(1, self.args.act_group_size//2)}], total_timesteps={self.total_timesteps}, abits={self.args.abits}, sym={self.args.a_sym}, static={self.args.static}'

def online_log_quantize(x: torch.Tensor, bits: int = 8, symmetric: bool = False):
    """
    对输入张量进行在线对数量化和反量化。
    通过平移处理负数，然后根据输入张量的最大值在线计算量化参数 (delta)。
    此版本支持分组量化，假设输入 x 是一个二维张量 [num_groups, group_size]，
    并沿 dim=1 对每个组进行量化。

    :param x: 输入张量
    :param bits: 量化位数
    :param symmetric: 是否使用对称量化
    :return: 伪量化后的张量
    """
    level = 2 ** bits
    if symmetric:
        # 对称量化范围 [-level/2, level/2 - 1]
        nb = -level // 2
        pb = level // 2 - 1
    else:
        # 非对称量化范围 [0, level - 1]
        nb = 0
        pb = level - 1

    # 增加平移项，将每组的最小值平移到0
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_shifted = x - x_min

    # 在线计算每组的量化参数 delta
    delta = x_shifted.max(dim=-1, keepdim=True)[0]

    # 为避免 log(0) 出现 NaN，将 x 中的 0 替换为一个极小值
    # torch.finfo(x.dtype).eps 是一个适合数据类型的很小的正数
    x_clipped = torch.clamp(x_shifted, min=torch.finfo(x.dtype).eps)

    # 对数量化
    # delta 为0时，x_clipped/delta 会是nan，需要处理
    delta = delta.clamp(min=1e-8)
    x_q = -1 * torch.log2(x_clipped / delta)
    x_q = torch.round(x_q)
    x_q = torch.clamp(x_q, nb, pb)

    # 反量化
    x_dq_shifted = (2 ** (-1 * x_q)) * delta

    # 将平移项加回去
    x_dq = x_dq_shifted + x_min

    return x_dq

class LogQuantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x


    def forward(self, hidden_states):
        if self.args.static == False or self.scales is None:
            savedShape = hidden_states.shape
            group_size = self.args.act_group_size
            if group_size > 0:
                hidden_states = hidden_states.reshape(-1, group_size)
                hidden_states = online_log_quantize(hidden_states, bits=self.args.abits, symmetric=self.args.a_sym)
                return hidden_states.reshape(savedShape)
            else:
                hidden_states = hidden_states.reshape(-1, savedShape[-1])
                return online_log_quantize(hidden_states, bits=self.args.abits, symmetric=self.args.a_sym).reshape(savedShape)
            
            

        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == False

        hidden_states = hidden_states.view(-1, savedShape[-1])
        selected_states = hidden_states.clone()

        if self.args.act_group_size > 0:
            selected_states = selected_states.reshape(-1, self.args.act_group_size)

        B, N, C = savedShape
        if self.args.act_group_size > 0:
            scales, base = self.scales[0].repeat(B * N, 1), self.scales[1].repeat(B * N, 1)
        else:
            scales, base = self.scales[0].unsqueeze(0).repeat(B * N, 1), self.scales[1].unsqueeze(0).repeat(B * N, 1)
        assert scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        selected_states = (torch.clamp(torch.round(selected_states / scales) + base, self.q_min, self.q_max) - base) * scales
        selected_states = selected_states.reshape(-1, savedShape[-1])
        hidden_states = selected_states
        
        return hidden_states.view(savedShape)
    
    def to(self, *args, **kwargs):
        super(LogQuantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def extra_repr(self):
        if self.args.static == True:
            return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, scale={self.scales[0]}, base={self.scales[1]}'
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'

