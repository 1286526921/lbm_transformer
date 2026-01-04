import numpy as np
import warnings

# -------------------------- LBM核心参数（D2Q9） --------------------------
Q = 9  # 离散速度方向数
# D2Q9离散速度（9个方向，2维）
e = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=np.float64)
# D2Q9权重系数（仅在守恒计算时使用）
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)

# -------------------------- 全局配置参数 --------------------------
SAMPLE_NUM = 120000  # 总样本数量
SAVE_PATH = "lbm_dataset_final_no_weight-range.npy"  # 保存路径
RHO_RANGE = (0.5, 1.5)  # 密度取值范围
PERTURB_SCALE = 0.01  # 基础扰动强度（无权重）
HIGH_ORDER_SCALE = 0.005  # 高阶矩扰动强度
MA_RANGE = (0.0, 0.3)  # 马赫数范围（替换原分层，改为连续范围）
c_s = 1.0 / np.sqrt(3)  # D2Q9声速（固定值）
c_s_sq = c_s ** 2  # 声速平方（预计算减少重复计算）
c_s_4 = c_s_sq ** 2  # 声速四次方（预计算）
MIN_F_EQ = 1e-8  # f_eq最小允许值（避免负数/零）
MAX_PERTURB_RATIO = 0.2  # 最大扰动比例（相对于f_eq，保证物理一致性）


# -------------------------- 平衡态计算（无权重+负数检测+非负修正） --------------------------
def compute_f_eq(rho, u, check_negative=True):
    """
    计算D2Q9无权重平衡态分布，新增负数检测和非负修正
    :param rho: 密度标量 (float64)
    :param u: 速度向量 (2,) (float64)
    :param check_negative: 是否检测并修正负数
    :return: f_eq: 无权重平衡态分布 (9,) (float64)
    """
    f_eq = np.zeros(Q, dtype=np.float64)
    u_sq = np.dot(u, u)  # 速度模长平方
    for i in range(Q):
        e_i = e[i]
        eu = np.dot(e_i, u)  # 速度方向与宏观速度的点积
        e_sq = np.dot(e_i, e_i)  # 离散速度模长平方

        # D2Q9无权重平衡态核心公式
        f_eq[i] = rho * (
                1 + eu / c_s_sq +
                (eu ** 2) / (2 * c_s_4) -
                u_sq / (2 * c_s_sq)  # 修正原公式错误：原e_sq→u_sq（D2Q9标准公式）
        )

    # 1. 检测负数情况
    negative_mask = f_eq < 0
    neg_count = np.sum(negative_mask)
    if neg_count > 0 and check_negative:
        warnings.warn(f"检测到{neg_count}个f_eq负数！rho={rho:.4f}, u={u}, f_eq={f_eq}")
        # 2. 非负修正（保留相对分布，保证守恒）
        f_eq = np.maximum(f_eq, MIN_F_EQ)
        # 修正后重新归一化（保证密度守恒）
        rho_calc = np.sum(w * f_eq)
        f_eq = f_eq * rho / rho_calc

    return f_eq


# -------------------------- 物理一致性扰动生成（替代原随机扰动） --------------------------
def generate_physically_consistent_perturbation(f_eq):
    """
    生成符合流体物理的扰动：
    1. 扰动幅度与f_eq正相关（避免小f_eq被过度扰动）
    2. 限制最大扰动比例（避免非物理的大扰动）
    3. 初始扰动保证非负性
    :param f_eq: 平衡态分布 (9,)
    :return: perturbation: 物理一致的初始扰动 (9,)
    """
    # 1. 生成与f_eq成比例的扰动幅度（自适应）
    perturb_amplitude = PERTURB_SCALE * f_eq
    # 2. 限制最大扰动比例（避免过度扰动）
    max_perturb = MAX_PERTURB_RATIO * f_eq
    perturb_amplitude = np.minimum(perturb_amplitude, max_perturb)

    # 3. 生成截断正态分布扰动（-1~1之间，避免极端值）
    perturbation = np.random.normal(0, 1, Q)
    perturbation = np.clip(perturbation, -1, 1)  # 截断到合理范围
    perturbation = perturbation * perturb_amplitude

    # 4. 预修正：保证扰动后f_non_eq非负（物理约束）
    min_allowed_perturb = MIN_F_EQ - f_eq
    perturbation = np.clip(perturbation, min_allowed_perturb, max_perturb)

    return perturbation


# -------------------------- 守恒扰动修正（显式补权重） --------------------------
def correct_perturbation(perturbation, f_eq):
    """
    修正扰动以保证密度/动量守恒，新增物理约束检查
    :param perturbation: 初始无权重扰动 (9,)
    :param f_eq: 平衡态分布（用于非负性验证）
    :return: corrected_pert: 守恒修正后的扰动 (9,)
    """
    # 1. 计算扰动导致的守恒偏差
    delta_rho = np.sum(w * perturbation)  # 密度偏差 = Σ(w·扰动)
    delta_mom = np.sum(w[:, np.newaxis] * perturbation[:, np.newaxis] * e, axis=0)  # 动量偏差

    # 2. 构造守恒约束矩阵
    constraint_matrix = np.vstack([
        w,  # 密度约束行
        w * e[:, 0],  # x动量约束行
        w * e[:, 1]  # y动量约束行
    ])

    # 3. 伪逆求解修正系数
    constraint_pinv = np.linalg.pinv(constraint_matrix)
    target_bias = np.array([delta_rho, delta_mom[0], delta_mom[1]], dtype=np.float64)
    correction_coeff = constraint_pinv @ target_bias

    # 4. 应用修正
    corrected_pert = perturbation - correction_coeff

    # 5. 验证修正后非负性（物理约束）
    f_non_eq_temp = f_eq + corrected_pert
    if np.any(f_non_eq_temp < MIN_F_EQ):
        # 二次修正：调整扰动幅度以保证非负
        scale_factor = 0.9
        corrected_pert = corrected_pert * scale_factor
        f_non_eq_temp = f_eq + corrected_pert
        if np.any(f_non_eq_temp < MIN_F_EQ):
            warnings.warn("修正后仍存在非负性问题，进一步缩小扰动幅度")
            corrected_pert = corrected_pert * 0.5

    # 验证修正效果
    delta_rho_new = np.sum(w * corrected_pert)
    delta_mom_new = np.sum(w[:, np.newaxis] * corrected_pert[:, np.newaxis] * e, axis=0)
    assert np.isclose(delta_rho_new, 0, atol=1e-10), f"密度修正失败: {delta_rho_new:.2e}"
    assert np.linalg.norm(delta_mom_new) < 1e-10, f"动量修正失败: {delta_mom_new}"

    return corrected_pert


# -------------------------- 高阶矩扰动（不破坏守恒） --------------------------
def add_high_order_perturbation(perturbation, f_eq):
    """
    添加动能矩高阶扰动，新增物理约束
    :param perturbation: 守恒修正后的扰动 (9,)
    :param f_eq: 平衡态分布（用于非负性验证）
    :return: final_pert: 含高阶矩偏差的扰动 (9,)
    """
    # 1. 构造正交于守恒约束的高阶矩基
    e_mag_sq = np.sum(e * e, axis=1)  # 离散速度模长平方 (9,)
    high_order_basis = w * (e_mag_sq - np.mean(e_mag_sq))

    # 正交化：移除与守恒基的投影
    high_order_basis = high_order_basis - np.dot(high_order_basis, w) * w / np.dot(w, w)
    wx = w * e[:, 0]
    high_order_basis = high_order_basis - np.dot(high_order_basis, wx) * wx / np.dot(wx, wx)
    wy = w * e[:, 1]
    high_order_basis = high_order_basis - np.dot(high_order_basis, wy) * wy / np.dot(wy, wy)

    # 归一化基向量
    basis_norm = np.linalg.norm(high_order_basis)
    if basis_norm > 1e-10:
        high_order_basis = high_order_basis / basis_norm

    # 2. 生成随机高阶矩扰动（限制幅度）
    high_order_pert = HIGH_ORDER_SCALE * np.random.normal(0, 1) * high_order_basis

    # 3. 验证不破坏守恒
    assert np.isclose(np.sum(w * high_order_pert), 0, atol=1e-10), "高阶矩扰动破坏密度守恒"
    assert np.linalg.norm(np.sum(w[:, np.newaxis] * high_order_pert[:, np.newaxis] * e, axis=0)) < 1e-10, \
        "高阶矩扰动破坏动量守恒"

    # 4. 叠加前验证非负性
    temp_pert = perturbation + high_order_pert
    f_non_eq_temp = f_eq + temp_pert
    if np.any(f_non_eq_temp < MIN_F_EQ):
        # 缩小高阶扰动幅度
        high_order_pert = high_order_pert * 0.5

    # 5. 叠加高阶矩扰动
    final_pert = perturbation + high_order_pert

    return final_pert


# -------------------------- 数据集生成主函数 --------------------------
def generate_enhanced_dataset():
    """
    生成无权重平衡态+显式补权重守恒的LBM数据集（马赫数范围采样+高阶矩+物理约束）
    """
    dataset = []
    # 统计信息
    stats = {
        "mach_numbers": [],
        "rho_values": [],
        "density_errors": [],
        "momentum_errors": [],
        "high_order_biases": [],
        "negative_f_eq_count": 0,  # f_eq负数次数
        "non_negative_correction_count": 0  # 非负修正次数
    }

    print(f"\n=== 按马赫数范围 [{MA_RANGE[0]}, {MA_RANGE[1]}] 生成 {SAMPLE_NUM} 个样本 ===")

    # 按范围生成所有样本（替换原分层逻辑）
    for sample_idx in range(SAMPLE_NUM):
        if sample_idx % 5000 == 0:  # 调整进度打印频率（避免过多输出）
            print(f"  进度: {sample_idx}/{SAMPLE_NUM}")

        # 从指定范围随机采样马赫数
        ma_actual = np.random.uniform(*MA_RANGE)
        max_u_mag = ma_actual * c_s  # 速度幅值（基于采样的马赫数）

        # 采样物理量
        rho = np.random.uniform(*RHO_RANGE)
        theta = np.random.uniform(0, 2 * np.pi)  # 随机速度方向
        u = np.array([
            max_u_mag * np.cos(theta),
            max_u_mag * np.sin(theta)
        ], dtype=np.float64)

        # 计算无权重平衡态（带负数检测）
        f_eq = compute_f_eq(rho, u)

        # 统计f_eq负数情况
        if np.sum(f_eq < 0) > 0:
            stats["negative_f_eq_count"] += 1

        # 生成物理一致性初始扰动
        perturbation = generate_physically_consistent_perturbation(f_eq)

        # 第一步修正：保证守恒
        perturb_corrected = correct_perturbation(perturbation, f_eq)

        # 第二步扩展：添加高阶矩扰动
        perturb_final = add_high_order_perturbation(perturb_corrected, f_eq)

        # 生成最终无权重非平衡态分布
        f_non_eq = f_eq + perturb_final

        # 最终非负性检查
        if np.any(f_non_eq < MIN_F_EQ):
            stats["non_negative_correction_count"] += 1
            f_non_eq = np.maximum(f_non_eq, MIN_F_EQ)
            # 重新修正守恒（保证物理约束）
            rho_calc = np.sum(w * f_non_eq)
            f_non_eq = f_non_eq * rho / rho_calc

        # 验证守恒性
        rho_calc = np.sum(w * f_non_eq)
        rho_true = np.sum(w * f_eq)
        density_error = abs(rho_calc - rho_true)

        mom_calc = np.sum(w[:, np.newaxis] * f_non_eq[:, np.newaxis] * e, axis=0)
        mom_true = np.sum(w[:, np.newaxis] * f_eq[:, np.newaxis] * e, axis=0)
        momentum_error = np.linalg.norm(mom_calc - mom_true)

        # 统计高阶矩偏差
        e_mag_sq = np.sum(e * e, axis=1)
        M_kin_eq = np.sum(w * f_eq * e_mag_sq)
        M_kin_non_eq = np.sum(w * f_non_eq * e_mag_sq)
        high_order_bias = abs(M_kin_non_eq - M_kin_eq)

        # 记录统计信息
        stats["mach_numbers"].append(ma_actual)
        stats["rho_values"].append(rho)
        stats["density_errors"].append(density_error)
        stats["momentum_errors"].append(momentum_error)
        stats["high_order_biases"].append(high_order_bias)

        # 添加到数据集
        dataset.append({
            "f_non_eq": f_non_eq,
            "f_eq": f_eq,
            "rho": rho,
            "u": u,
            "mach_number": ma_actual,
            "density_error": density_error,
            "momentum_error": momentum_error,
            "high_order_bias": high_order_bias
        })

    # 保存数据集
    np.save(SAVE_PATH, dataset)

    # 输出统计报告
    print("\n=== 数据集生成完成（物理约束增强版） ===")
    print(f"📊 核心统计：")
    print(f"  总样本数: {len(dataset)}")
    print(f"  马赫数范围: {np.min(stats['mach_numbers']):.3f} ~ {np.max(stats['mach_numbers']):.3f}")
    print(f"  密度范围: {np.min(stats['rho_values']):.3f} ~ {np.max(stats['rho_values']):.3f}")
    print(f"  平均密度守恒误差: {np.mean(stats['density_errors']):.2e} (最大: {np.max(stats['density_errors']):.2e})")
    print(f"  平均动量守恒误差: {np.mean(stats['momentum_errors']):.2e} (最大: {np.max(stats['momentum_errors']):.2e})")
    print(
        f"  平均高阶矩偏差: {np.mean(stats['high_order_biases']):.2e} (最大: {np.max(stats['high_order_biases']):.2e})")
    print(f"  🔍 物理约束统计：")
    print(f"    f_eq负数出现次数: {stats['negative_f_eq_count']}")
    print(f"    非负性修正次数: {stats['non_negative_correction_count']}")
    print(f"💾 保存路径: {SAVE_PATH}")

    # 验证加载
    loaded_dataset = np.load(SAVE_PATH, allow_pickle=True)
    print(loaded_dataset[0]['f_non_eq'].dtype)
    print(f"\n✅ 加载验证：")
    print(f"  加载样本数: {len(loaded_dataset)}")
    print(f"  第一个样本马赫数: {loaded_dataset[0]['mach_number']:.3f}")
    print(f"  第一个样本密度误差: {loaded_dataset[0]['density_error']:.2e}")
    print(f"  第一个样本f_eq最小值: {np.min(loaded_dataset[0]['f_eq']):.4e}")
    print(f"  第一个样本f_non_eq最小值: {np.min(loaded_dataset[0]['f_non_eq']):.4e}")

    return dataset


# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 生成最终版数据集
    generate_enhanced_dataset()