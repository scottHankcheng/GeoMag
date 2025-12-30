import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from itertools import repeat
import collections.abc
import math

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable): return x
    return tuple(repeat(x, 2))

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0: random_tensor.div_(keep_prob)
        return x * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features or in_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=192, embed_dim=192, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x, hw_shape):
        B, L, C = x.shape; H, W = hw_shape
        return x.transpose(1, 2).view(B, self.embed_dim, H, W)

class MambaEquivalentBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, d_state=16, d_conv=4, expand=2, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, bias=False)
    def forward(self, x):
        return x + self.conv2(self.pad2(self.relu(self.conv1(self.pad1(x)))))

class Manipulator(nn.Module):
    def __init__(self, num_resblk, embed_dim):
        super().__init__()
        self.convblks = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(embed_dim, embed_dim, 7, 1, bias=False),
            nn.ReLU(True)
        )
        self.convblks_after = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, bias=False)
        )
        self.resblks = nn.ModuleList([ResBlock(embed_dim) for _ in range(num_resblk)])

    def forward(self, x_a, x_b, amp):
        diff = x_b - x_a
        diff = self.convblks(diff)
        
        diff = (amp - 1.0) * diff 
        
        diff = self.convblks_after(diff)
        for blk in self.resblks:
            diff = blk(diff)
        return x_b + diff

class GeoMag(nn.Module):
    def __init__(self, img_size=384, in_chans=3, embed_dim=192, depth=12,
                 mlp_ratio=2.0, drop_rate=0., drop_path_rate=0.1, d_state=16, d_conv=4, expand=2,
                 manipulator_num_resblk=1, use_checkpoint=False, img_range=1., **kwargs):
        super().__init__()
        
        if Mamba is None: raise ImportError("mamba_ssm is not installed.")
        
        self.use_checkpoint = use_checkpoint
        self.downsample_rate = 8

        self.conv_shallow = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        shallow_mid_dim = embed_dim // 2
        
        self.shallow_diff_proc = nn.Sequential(
            nn.Conv2d(embed_dim, shallow_mid_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(shallow_mid_dim, embed_dim, 1, 1, 0)
        )
        self.static_mask_gen = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
            nn.Sigmoid() 
        )
        self.static_processor = nn.Sequential(
            ResBlock(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        )
        
        self.conv_first = nn.Conv2d(embed_dim, embed_dim, self.downsample_rate, self.downsample_rate, 0)

        hw_after_conv_first = img_size // self.downsample_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        mamba_block_kwargs = {'mlp_ratio': mlp_ratio, 'd_state': d_state, 'd_conv': d_conv, 'expand': expand}
        norm_layer = nn.LayerNorm
        num_patches = hw_after_conv_first * hw_after_conv_first

        self.patch_embed_dfe = PatchEmbed(embed_dim, embed_dim, norm_layer)
        self.absolute_pos_embed_dfe = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop_dfe = nn.Dropout(drop_rate)
        self.encoder_dfe = nn.ModuleList([MambaEquivalentBlock(embed_dim, drop_path=dpr[i], **mamba_block_kwargs) for i in range(depth)])
        self.norm_dfe = norm_layer(embed_dim)
        self.conv_after_body_dfe = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.manipulator = Manipulator(manipulator_num_resblk, embed_dim)
        
        self.patch_embed_mmsa = PatchEmbed(embed_dim, embed_dim, norm_layer)
        self.absolute_pos_embed_mmsa = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop_mmsa = nn.Dropout(drop_rate)
        self.encoder_mmsa = nn.ModuleList([MambaEquivalentBlock(embed_dim, drop_path=dpr[i], **mamba_block_kwargs) for i in range(depth)])
        self.norm_mmsa = norm_layer(embed_dim)
        self.conv_after_body_mmsa = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * self.downsample_rate**2, 3, 1, 1),
            nn.PixelShuffle(self.downsample_rate)
        )
        
        self.fusion_block = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1), 
            nn.GroupNorm(num_groups=32, num_channels=embed_dim),
            ResBlock(embed_dim),
            nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        )
        self.patch_unembed = PatchUnEmbed(embed_dim)
        
        torch.nn.init.trunc_normal_(self.absolute_pos_embed_dfe, std=.02)
        torch.nn.init.trunc_normal_(self.absolute_pos_embed_mmsa, std=.02)
        self.apply(self._init_weights)
        
        nn.init.constant_(self.static_mask_gen[0].bias, 2.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _interpolate_pos_encoding(self, x, p, h):
        B, L, C = x.shape; L_base = p.shape[1]
        if L == L_base: return x + p
        H_base = W_base = int(math.sqrt(L_base))
        if H_base * W_base != L_base: return x
        p2d = p.reshape(1, H_base, W_base, C).permute(0, 3, 1, 2)
        p_int = F.interpolate(p2d, size=h, mode='bicubic', align_corners=False)
        return x + p_int.permute(0, 2, 3, 1).reshape(1, L, C)

    def forward_branch(self, x, patch_embed, pos_embed, pos_drop, encoder, norm, conv_after_body):
        hw = (x.shape[2], x.shape[3])
        x_seq = patch_embed(x)
        x_seq = self._interpolate_pos_encoding(x_seq, pos_embed, hw)
        x_seq = pos_drop(x_seq)
        for blk in encoder:
            x_seq = checkpoint.checkpoint(blk, x_seq) if self.use_checkpoint and self.training else blk(x_seq)
        x = self.patch_unembed(norm(x_seq), hw)
        return conv_after_body(x)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        rate = self.downsample_rate
        mod_pad_h = (rate - h % rate) % rate
        mod_pad_w = (rate - w % rate) % rate
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, a, b, amp, c=None):
        original_H, original_W = a.shape[2], a.shape[3]
        a = self.check_image_size(a)
        b = self.check_image_size(b)
        
        a_shallow_feat = self.conv_shallow(a)
        b_shallow_feat = self.conv_shallow(b)

        diff = b_shallow_feat - a_shallow_feat
        
        diff_abs = torch.clamp(diff.abs(), min=0.0, max=1.0) 

        diff_feat = self.shallow_diff_proc(diff_abs) 
        static_mask = 1.0 - self.static_mask_gen(diff_feat)
        
        masked_static_feat = a_shallow_feat * static_mask 
        
        refined_output = self.static_processor(masked_static_feat)
        
        a_final_for_fusion = a_shallow_feat + refined_output

        a_first = self.conv_first(a_shallow_feat)
        b_first = self.conv_first(b_shallow_feat)

        res_a = self.forward_branch(a_first, self.patch_embed_dfe, self.absolute_pos_embed_dfe, self.pos_drop_dfe, self.encoder_dfe, self.norm_dfe, self.conv_after_body_dfe) + a_first
        res_b = self.forward_branch(b_first, self.patch_embed_dfe, self.absolute_pos_embed_dfe, self.pos_drop_dfe, self.encoder_dfe, self.norm_dfe, self.conv_after_body_dfe) + b_first
        
        m = self.manipulator(res_a, res_b, amp)
        res_m = self.forward_branch(m, self.patch_embed_mmsa, self.absolute_pos_embed_mmsa, self.pos_drop_mmsa, self.encoder_mmsa, self.norm_mmsa, self.conv_after_body_mmsa) + m
        
        upsampled_m = self.upsampler(res_m)
        
        a_final_for_fusion = torch.clamp(a_final_for_fusion, min=-2.0, max=2.0)
        
        fused_feat = torch.cat([upsampled_m, a_final_for_fusion], dim=1)
        
        y_hat_padded = self.fusion_block(fused_feat)
        y_hat_padded = torch.tanh(y_hat_padded)
        
        y_hat = y_hat_padded[:, :, :original_H, :original_W]
        
        res_c = None
        if c is not None and self.training:
            c = self.check_image_size(c)
            c_shallow_feat = self.conv_shallow(c)
            c_first = self.conv_first(c_shallow_feat)
            res_c = self.forward_branch(c_first, self.patch_embed_dfe, self.absolute_pos_embed_dfe, self.pos_drop_dfe, self.encoder_dfe, self.norm_dfe, self.conv_after_body_dfe) + c_first
            
        return y_hat, res_a, res_b, res_c


