import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from lbm_transformer import LBMTrajectoryTransformer, physical_constraints



# ===================== 核心修复：设备与路径优化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_FILE = "latest_checkpoint.pth"  # 移除文件夹，直接保存到当前目录
TRAIN_LOG_FILE = "train_log.json"
BEST_CHECKPOINT_FILE = "best_checkpoint.pth"

print(f"使用设备：{device}")


# ===================== 数据集类（适配新数据集格式） =====================
class LBMDataset(Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在：{data_path}")

        data = np.load(data_path, allow_pickle=True)
        self.samples = []
        for item in data:
            # 从数据集中直接获取预计算的物理量
            f_input = item["f_non_eq"]       # 非平衡态分布（输入）
            f_eq_target = item["f_eq"]       # 平衡态分布（目标）
            rho = item["rho"]                # 真实密度（直接使用，无需计算）
            u = item["u"]                    # 真实速度（直接使用，无需计算）
            self.samples.append((f_input, f_eq_target, rho, u))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_in, f_eq, rho, u = self.samples[idx]
        return (
            torch.tensor(f_in, dtype=torch.float32),
            torch.tensor(f_eq, dtype=torch.float32),
            torch.tensor(rho, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32)
        )


# ===================== 断点保存：增强版（原子操作） =====================
def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_loss):
    try:
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'avg_loss': float(avg_loss),
            'best_loss': float(best_loss)
        }

        # 原子操作保存：先写临时文件，再重命名
        temp_checkpoint = CHECKPOINT_FILE + ".tmp"
        torch.save(checkpoint, temp_checkpoint)
        os.replace(temp_checkpoint, CHECKPOINT_FILE)  # 原子操作替换

        if avg_loss < best_loss:
            temp_best = BEST_CHECKPOINT_FILE + ".tmp"
            torch.save(checkpoint, temp_best)
            os.replace(temp_best, BEST_CHECKPOINT_FILE)  # 原子操作替换
            print(f"✅ 最佳模型已备份：{BEST_CHECKPOINT_FILE}")

        log_data = {
            'current_epoch': int(epoch),
            'best_loss': float(best_loss),
            'last_loss': float(avg_loss),
            'save_time': str(os.popen('date').read().strip())
        }
        # 日志文件也使用原子操作
        temp_log = TRAIN_LOG_FILE + ".tmp"
        with open(temp_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        os.replace(temp_log, TRAIN_LOG_FILE)

        print(f"✅ 断点已保存：第{epoch}轮，损失={avg_loss:.6f}")

    except PermissionError:
        print(f"❌ 权限错误：无法写入文件 {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"❌ 保存断点失败：{str(e)}")


# ===================== 断点加载：增强版 =====================
def load_checkpoint(model, optimizer, scheduler):
    if not os.path.exists(CHECKPOINT_FILE):
        print("⚠️ 未找到断点文件，将从头开始训练")
        return False, 0, float('inf')

    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️ 优化器状态加载失败：{str(e)}")

        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️ 调度器状态加载失败：{str(e)}")

        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        best_loss = float(checkpoint.get('best_loss', float('inf')))
        last_loss = float(checkpoint.get('avg_loss', 0.0))

        print(f"✅ 成功加载断点：从第{start_epoch}轮开始训练（上一轮损失={last_loss:.6f}，最佳损失={best_loss:.6f}）")
        return True, start_epoch, best_loss

    except RuntimeError as e:
        print(f"❌ 模型参数不匹配：{str(e)}，将从头开始训练")
        return False, 0, float('inf')
    except Exception as e:
        print(f"❌ 断点文件损坏：{str(e)}，将从头开始训练")
        return False, 0, float('inf')


# ===================== 训练函数 =====================
def train():
    # 基础配置（更新数据集路径为lbm_dataset.py生成的路径）
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    total_epochs = 5528
    lr = 1e-4
    data_path = "lbm_dataset_final_no_weight.npy"  # 匹配lbm_dataset.py的SAVE_PATH

    # 数据加载
    try:
        dataset = LBMDataset(data_path)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"❌ 数据集加载失败：{str(e)}")
        return

    # 模型初始化
    model = LBMTrajectoryTransformer().to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"✅ 使用 {torch.cuda.device_count()} 个GPU训练")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # 加载断点
    loaded, start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler)

    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (f_in, f_eq_target, rho, u) in enumerate(train_loader):
            f_in = f_in.unsqueeze(1).to(device)  # (batch,1,9)
            f_eq_target = f_eq_target.to(device)
            rho = rho.to(device)
            u = u.to(device)

            # 前向传播
            f_eq_pred = model(f_in)

            e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                          [1, 1], [-1, 1], [-1, -1], [1, -1]],dtype=np.float64)  # (9,2) 离散速度方向
            # 在损失计算前定义D2Q9权重（与lbm_dataset.py保持一致）
            w = torch.tensor([
                4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
                1 / 36, 1 / 36, 1 / 36, 1 / 36
            ], device=device, dtype=torch.float32)  # 确保与模型在同一设备

            # 损失计算（增加权重）
            # 1. 加权MSE损失：对每个速度方向的误差按权重w加权
            weighted_pred = f_eq_pred * w  # 预测值加权
            weighted_target = f_eq_target * w  # 目标值加权
            mse_loss = F.mse_loss(weighted_pred, weighted_target)

            # 2. 物理约束损失（保持不变，已内置权重）
            phys_loss = physical_constraints(f_eq_pred, rho, u, e)

            # 3. 正则损失（保持不变）
            reg_loss = sum(p.pow(2).sum() for p in model.parameters()) * 1e-5

            # 总损失
            loss = mse_loss + phys_loss + reg_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 更新学习率
        scheduler.step()

        # 更新最佳损失
        current_best_loss = min(best_loss, avg_loss)

        # 保存断点
        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, current_best_loss)
        best_loss = current_best_loss

        # 打印日志
        print(
            f"📌 Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6e}")

    # 训练完成：保存最终模型
    try:
        final_model_path = "lbm_transformer_final.pth"
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        # 最终模型也使用原子操作保存
        temp_final = final_model_path + ".tmp"
        torch.save(model_state, temp_final)
        os.replace(temp_final, final_model_path)
        print(f"🎉 训练完成！最终模型已保存为 {final_model_path}")
    except Exception as e:
        print(f"❌ 保存最终模型失败：{str(e)}")


if __name__ == "__main__":
    train()