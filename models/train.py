import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
from models.dir_aware_det import DirAwareDetector
from utils.loss import AdaptiveWeightedLoss
from utils.metrics import compute_map
from utils.setup import set_random_seed
from data.munich_dataset import MunichDataset
from data.uav_mato_dataset import UAVMATODataset

def train(args):
    # 1. 配置加载与初始化
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    set_random_seed(config['seed'])  # 固定随机种子=42（论文设置）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 数据集与加载器
    if config['dataset'] == 'munich':
        train_dataset = MunichDataset(config['data_root'], split='train', augment=True)
        val_dataset = MunichDataset(config['data_root'], split='val', augment=False)
    elif config['dataset'] == 'uav_mato':
        train_dataset = UAVMATODataset(config['data_root'], split='train', augment=True)
        val_dataset = UAVMATODataset(config['data_root'], split='val', augment=False)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['val_batch_size'], shuffle=False, num_workers=4
    )
    
    # 3. 模型、损失、优化器
    model = DirAwareDetector(
        num_classes=config['num_classes'],
        stack_depth=config['stack_depth']
    ).to(device)
    criterion = AdaptiveWeightedLoss(num_classes=config['num_classes']).to(device)
    # 优化器：主模型参数 + 损失平衡参数δ（论文设置Adam，lr分别为1e-3和1e-4）
    params = [
        {'params': model.parameters(), 'lr': config['lr'], 'weight_decay': config['weight_decay']},
        {'params': criterion.parameters(), 'lr': config['loss_lr'], 'weight_decay': 0}
    ]
    optimizer = optim.Adam(params, betas=(0.9, 0.999))
    # 学习率调度：余弦退火（论文设置周期5000迭代）
    scheduler = CosineAnnealingLR(optimizer, T_max=config['anneal_cycle'], eta_min=config['min_lr'])
    
    # 4. 训练循环
    best_val_map = 0.0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            # 目标移到设备
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # 前向传播
            preds = model(imgs)
            loss, loss_dict = criterion(preds, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 日志记录
            total_loss += loss.item()
            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Avg Loss: {avg_loss:.4f}, Cls Loss: {loss_dict['cls_loss']:.4f}, "
                      f"Angle Loss: {loss_dict['angle_loss']:.4f}")
        
        # 5. 验证
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                val_preds.append(preds)
                val_targets.append(targets)
        # 计算mAP（论文指标）
        val_map = compute_map(val_preds, val_targets, config['iou_threshold'])
        print(f"Epoch [{epoch+1}/{config['epochs']}], Val mAP: {val_map:.4f}")
        
        # 保存最优模型
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), config['save_path'] + f"best_model_epoch{epoch+1}.pth")
            print(f"Best model saved (Val mAP: {best_val_map:.4f})")
    
    print(f"Training finished! Best Val mAP: {best_val_map:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DirAware-LightDet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (e.g., configs/munich.yaml)')
    args = parser.parse_args()
    train(args)
