import torch
import torch.nn as nn
import torch.nn.functional as F


class LBMTrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        # 输入投影（无偏置）
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # 残差适配层：匹配输入维度到hidden_dim
        self.residual_adapter = nn.Linear(input_dim, hidden_dim,
                                          bias=False) if input_dim != hidden_dim else nn.Identity()

        # 单节点位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer编码器（无偏置）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            bias=False,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影（无偏置）
        self.output_proj = nn.Linear(hidden_dim, input_dim, bias=False)

        # 输出残差适配层：匹配hidden_dim到input_dim
        self.output_residual_adapter = nn.Linear(hidden_dim, input_dim,
                                                 bias=False) if hidden_dim != input_dim else nn.Identity()

        # D2Q9固定参数（内置，避免重复定义）
        self.register_buffer('w', torch.tensor([
            4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
            1 / 36, 1 / 36, 1 / 36, 1 / 36
        ], dtype=torch.float32))  # (9,)

        self.register_buffer('e', torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ], dtype=torch.float32))  # (9,2)

    def forward(self, x):
        # x shape: (batch_size, 1, 9) 单节点输入（f_non_eq/fi）
        batch_size = x.shape[0]
        x_original = x  # 保存原始输入用于归一化基准
        x_flat = x_original.squeeze(1)  # (batch, 9)

        # ========== 原有特征提取逻辑（保留残差连接） ==========
        # 输入投影 + ReLU + 位置编码 + 输入残差
        x_proj = self.input_proj(x)  # (batch, 1, hidden)
        x_proj = F.relu(x_proj)
        x_residual = self.residual_adapter(x_original)  # 维度适配
        x = x_proj + x_residual + self.pos_encoding

        # Transformer处理 + 核心残差
        x_transformer = self.transformer(x)  # (batch, 1, hidden)
        x_transformer = F.relu(x_transformer)
        x = x + x_transformer

        # 输出投影 + 输出残差
        f_eq_pred = self.output_proj(x)  # (batch, 1, 9)
        x_output_residual = self.output_residual_adapter(x)
        f_eq_pred = f_eq_pred + x_output_residual
        f_eq_pred = f_eq_pred.squeeze(1)  # (batch, 9)

        # ========== 密度归一化 + 动量修正（集成到forward） ==========
        # 1. 密度归一化（以输入fi为基准）
        # 计算输入fi的密度（基准密度）
        rho_i = torch.sum(x_flat * self.w, dim=1, keepdim=True)  # (batch, 1)
        # 计算预测值的密度
        rho_pred = torch.sum(f_eq_pred * self.w, dim=1, keepdim=True)  # (batch, 1)
        # 密度归一化系数（避免除零）
        alpha = rho_i / (rho_pred + 1e-8)
        # 应用密度归一化
        f_eq_denorm = f_eq_pred * alpha  # (batch, 9)

        # 2. 动量修正（以输入fi的动量为基准）
        # 计算输入fi的动量（基准动量）
        j_i = torch.einsum('bd,dk->bk', x_flat * self.w.unsqueeze(0), self.e)  # (batch, 2)
        # 计算密度归一化后的动量
        j_denorm = torch.einsum('bd,dk->bk', f_eq_denorm * self.w.unsqueeze(0), self.e)  # (batch, 2)
        # 动量残差
        delta_j = j_i - j_denorm  # (batch, 2)
        # 计算动量修正项（D2Q9最优修正公式）
        delta_f = self.w.unsqueeze(0) * (9 / 2) * torch.einsum('bk,dk->bd', delta_j, self.e)  # (batch, 9)
        # 最终修正后的分布（满足密度+动量双守恒）
        f_eq_fin = f_eq_denorm + delta_f

        return f_eq_fin  # 直接返回满足双守恒的预测结果


def physical_constraints(f_pred, rho=None, u=None, e=None):
    """物理约束损失项（仅计算损失，归一化/修正已在forward完成）"""
    loss = 0.0

    # D2Q9权重系数（与模型内置一致）
    w = torch.tensor([
        4 / 9,  # (0,0)
        1 / 9,  # (1,0)
        1 / 9,  # (0,1)
        1 / 9,  # (-1,0)
        1 / 9,  # (0,-1)
        1 / 36,  # (1,1)
        1 / 36,  # (-1,1)
        1 / 36,  # (-1,-1)
        1 / 36  # (1,-1)
    ], device=f_pred.device, dtype=torch.float32)  # (9,)

    # 1. 密度约束损失（监控归一化效果）
    if rho is not None:
        rho_pred = torch.sum(f_pred * w, dim=1)  # (batch,)
        loss += F.mse_loss(rho_pred, rho) * 10.0

    # 2. 动量约束损失（监控修正效果）
    if u is not None and e is not None:
        e = torch.tensor(e, device=f_pred.device, dtype=torch.float32)  # (9,2)
        weighted_f = f_pred * w.unsqueeze(0)  # (batch,9)
        momentum_pred = torch.einsum('bd,dk->bk', weighted_f, e)  # (batch,2)
        momentum_target = rho.unsqueeze(1) * u  # (batch,2)
        loss += F.mse_loss(momentum_pred, momentum_target) * 10.0

    return loss  # 仅返回损失，无需返回修正后的f_pred