import numpy as np

# D2Q9参数
e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
              [1, 1], [-1, 1], [-1, -1], [1, -1]])  # (9,2) 离散速度方向
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # (9,) 权重
c_s = 1 / np.sqrt(3)  # 声速


def compute_f_eq(ρ, u, weighted=False):
    """
    支持批量计算平衡分布函数（D2Q9）
    ρ: (NX, NY) 或标量，密度场
    u: (NX, NY, 2) 或(2,)，速度场
    weighted: 是否带权重计算
    返回: (NX, NY, 9) 或(9,)，平衡态分布函数
    """
    if len(u.shape) == 1:  # 单个格子
        u = u.reshape(1, 1, 2)
        ρ = np.array(ρ).reshape(1, 1)

    NX, NY = ρ.shape
    f_eq = np.zeros((NX, NY, 9))

    e_dot_u = np.einsum('kd,ijd->ijk', e, u)  # (NX, NY, 9)
    u_sq = np.sum(u **2, axis=2, keepdims=True)  # (NX, NY, 1)

    for d in range(9):
        term = 1 + 3 * e_dot_u[:, :, d] + 4.5 * (e_dot_u[:, :, d]** 2) - 1.5 * u_sq[:, :, 0]
        f_eq[:, :, d] = ρ * term
        if weighted:
            f_eq[:, :, d] *= w[d]  # 带权重版本

    return f_eq[0, 0] if (NX == 1 and NY == 1) else f_eq




def zou_he_velocity_boundary(f, boundary_mask, boundary_u):
    """
    应用Zou-He速度边界条件（支持无滑移/指定速度，修正角落处理）
    参数:
        f: (NX, NY, 9)，碰撞后的分布函数
        boundary_mask: (NX, NY)，True表示边界格子
        boundary_u: (2,) 或 (NX, NY, 2)，边界格子的指定速度（如无滑移为(0,0)）
    返回:
        f: 处理边界后的分布函数
    """
    f = f.copy()
    NX, NY = boundary_mask.shape
    # 统一边界速度格式为(NX, NY, 2)
    if len(np.array(boundary_u).shape) == 1:
        boundary_u = np.tile(boundary_u, (NX, NY, 1))

    for i in range(NX):
        for j in range(NY):
            if boundary_mask[i, j]:
                u_wall = boundary_u[i, j]  # 边界速度 (u_x, u_y)
                f0, f1, f2, f3, f4, f5, f6, f7, f8 = f[i, j]

                # ----------------- 角落边界：优先处理（i=0且j=0等）-----------------
                if i == 0 and j == 0:  # 左下角
                    ρ = (f0 + 2*(f3 + f4 + f6 + f7)) / (1 - u_wall[0] - u_wall[1])
                    f[i,j,1] = f3 + 2/3*ρ*u_wall[0]
                    f[i,j,2] = f4 + 2/3*ρ*u_wall[1]
                    f[i,j,5] = 0.5*(f[i,j,1] + f[i,j,2] - f0) + 0.5*ρ*(u_wall[0]+u_wall[1])
                elif i == 0 and j == NY-1:  # 左上角
                    ρ = (f0 + 2*(f3 + f2 + f6 + f5)) / (1 - u_wall[0] + u_wall[1])
                    f[i,j,1] = f3 + 2/3*ρ*u_wall[0]
                    f[i,j,4] = f2 - 2/3*ρ*u_wall[1]
                    f[i,j,8] = 0.5*(f[i,j,1] + f[i,j,4] - f0) + 0.5*ρ*(u_wall[0]-u_wall[1])
                elif i == NX-1 and j == 0:  # 右下角
                    ρ = (f0 + 2*(f1 + f4 + f5 + f8)) / (1 + u_wall[0] - u_wall[1])
                    f[i,j,3] = f1 - 2/3*ρ*u_wall[0]
                    f[i,j,2] = f4 + 2/3*ρ*u_wall[1]
                    f[i,j,7] = 0.5*(f[i,j,3] + f[i,j,2] - f0) + 0.5*ρ*(-u_wall[0]+u_wall[1])
                elif i == NX-1 and j == NY-1:  # 右上角
                    ρ = (f0 + 2*(f1 + f2 + f8 + f5)) / (1 + u_wall[0] + u_wall[1])
                    f[i,j,3] = f1 - 2/3*ρ*u_wall[0]
                    f[i,j,4] = f2 - 2/3*ρ*u_wall[1]
                    f[i,j,7] = 0.5*(f[i,j,3] + f[i,j,4] - f0) + 0.5*ρ*(-u_wall[0]-u_wall[1])

                # ----------------- 边边界（非角落）-----------------
                elif i == 0:  # 左边界
                    ρ = (f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / (1 - u_wall[0])
                    f[i,j,1] = f3 + 2/3*ρ*u_wall[0]
                    f[i,j,5] = f7 + 1/3*ρ*u_wall[0] + 0.5*ρ*u_wall[1]
                    f[i,j,8] = f6 + 1/3*ρ*u_wall[0] - 0.5*ρ*u_wall[1]
                elif i == NX-1:  # 右边界
                    ρ = (f0 + f2 + f4 + 2 * (f1 + f5 + f8)) / (1 + u_wall[0])
                    f[i,j,3] = f1 - 2/3*ρ*u_wall[0]
                    f[i,j,6] = f5 - 1/3*ρ*u_wall[0] + 0.5*ρ*u_wall[1]
                    f[i,j,7] = f8 - 1/3*ρ*u_wall[0] - 0.5*ρ*u_wall[1]
                elif j == 0:  # 下边界
                    ρ = (f0 + f1 + f3 + 2 * (f4 + f7 + f8)) / (1 - u_wall[1])
                    f[i,j,2] = f4 + 2/3*ρ*u_wall[1]
                    f[i,j,5] = f8 + 0.5*ρ*u_wall[0] + 1/3*ρ*u_wall[1]
                    f[i,j,6] = f7 - 0.5*ρ*u_wall[0] + 1/3*ρ*u_wall[1]
                elif j == NY-1:  # 上边界
                    ρ = (f0 + f1 + f3 + 2 * (f2 + f5 + f6)) / (1 + u_wall[1])
                    f[i,j,4] = f2 - 2/3*ρ*u_wall[1]
                    f[i,j,7] = f6 - 0.5*ρ*u_wall[0] - 1/3*ρ*u_wall[1]
                    f[i,j,8] = f5 + 0.5*ρ*u_wall[0] - 1/3*ρ*u_wall[1]

    return f


def lbm_update_3x3(grid_ρ, grid_u, boundary_mask=None, boundary_u=(0, 0), τ=0.53, f_old=None):
    """
    3x3网格的LBM更新（BGK模型+Zou-He速度边界，修正f_old逻辑）
    参数:
        grid_ρ: (3,3)，当前密度场
        grid_u: (3,3,2)，当前速度场
        boundary_mask: (3,3)，边界掩码（True=固体/速度边界）
        boundary_u: (2,)，边界格子的指定速度（无滑移默认(0,0)）
        τ: 弛豫时间（默认0.53，τ>0.5稳定）
        f_old: (3,3,9)，上一步迁移后的分布函数（迭代时必须传入）
    返回:
        ρ_new: 中心格子(1,1)的新密度
        u_new: 中心格子(1,1)的新速度
        f_mig: 整个3x3网格迁移后的分布函数（下次迭代用f_old）
    """
    # 1. 初始分布函数：迭代时用f_old，首次用平衡态
    if f_old is None:
        f_old = compute_f_eq(grid_ρ, grid_u)

    # 2. 计算平衡态分布函数
    f_eq = compute_f_eq(grid_ρ, grid_u)

    # 3. BGK碰撞
    f_collide = f_old + (f_eq - f_old) / τ

    # 4. 应用Zou-He速度边界
    if boundary_mask is not None:
        f_collide = zou_he_velocity_boundary(f_collide, boundary_mask, boundary_u)

    # 5. 修正：整个3x3网格的迁移（而非仅中心格子）
    NX, NY = grid_ρ.shape
    f_mig = np.zeros_like(f_collide)
    for i in range(NX):
        for j in range(NY):
            for d in range(9):
                # 迁移：粒子从 (i-e[d][0], j-e[d][1]) 到 (i,j)
                src_i = i - e[d][0]
                src_j = j - e[d][1]
                # 周期性边界（3x3网格若需周期，否则用边界条件）
                src_i = np.clip(src_i, 0, NX-1)
                src_j = np.clip(src_j, 0, NY-1)
                f_mig[i, j, d] = f_collide[src_i, src_j, d]

    # 6. 计算中心格子的新密度和速度
    center_i, center_j = 1, 1
    ρ_new = np.sum(f_mig[center_i, center_j])
    u_new = np.sum(f_mig[center_i, center_j].reshape(-1, 1) * e, axis=0) / ρ_new if ρ_new > 1e-6 else np.zeros(2)

    return ρ_new, u_new