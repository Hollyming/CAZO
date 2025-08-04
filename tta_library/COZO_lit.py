import torch
import torch.nn as nn
import cma
import numpy as np
import time
import os
from models.adaformer import AdaFormerViT

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *

RUNNING_IMAGNET_R = False


class Optimizer:
    """优化器基类"""
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, grad_estimate):
        """更新参数"""
        raise NotImplementedError

class SGD(Optimizer):
    """SGD优化器"""
    def __init__(self, lr):
        super().__init__(lr)
    
    def step(self, grad_estimate):
        return -self.lr * grad_estimate     #w-lr*g

class SGD_Momentum(Optimizer):
    """带动量的SGD优化器"""
    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.momentum = None
    
    def step(self, grad_estimate):
        if self.momentum is None:
            self.momentum = (1 - self.beta) * grad_estimate
        else:
            self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_estimate
        return -self.lr * self.momentum

    def reset(self):
        """重置动量"""
        self.momentum = None

class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
    
    def step(self, grad_estimate):
        self.t += 1
        if self.m is None:
            self.m = torch.zeros_like(grad_estimate)
            self.v = torch.zeros_like(grad_estimate)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_estimate ** 2
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0


class COZO_Ablation(nn.Module):
    """COZO算法的消融实验版本，用于分析均值m和协方差C的贡献"""
    def __init__(self, model: AdaFormerViT, mode='full', fitness_lambda=0.4, lr=0.01, 
                 pertub=14, perturbation_scale=10.0, optimizer_type='sgd', beta=0.9):
        """
        Args:
            model: AdaFormerViT模型
            mode: 'full' - 完整COZO (N(m,C))
                 'mean_only' - 只使用均值 (m + N(0,1))
                 'cov_only' - 只使用协方差 (N(0,C))
            fitness_lambda: 适应度函数的平衡因子
            lr: 学习率
            pertub: 扰动数量
            perturbation_scale: 扰动缩放因子
            optimizer_type: 优化器类型，可选'sgd'或'sgd_momentum'
            beta: 动量系数，仅当optimizer_type='sgd_momentum'时有效
        """
        super().__init__()
        self.mode = mode
        self.fitness_lambda = fitness_lambda
        self.lr = lr
        self.zo_eps = 1.0
        self.pertub = pertub - 1
        # 扰动缩减因子,用于控制扰动大小
        self.perturbation_scale = perturbation_scale
        
        self.model = model
        # 确保所有adapter参数不需要梯度
        for adapter in self.model.adapters.values():
            for param in adapter.parameters():
                param.requires_grad_(False)
                
        self.es = self._init_cma()

        # 保存最佳adapter参数
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        self.best_loss = np.inf
        self.final_loss = np.inf
        self.hist_stat = None

        # 初始化优化器
        if optimizer_type == 'sgd':
            self.optimizer = SGD(lr)
        elif optimizer_type == 'sgd_momentum':
            self.optimizer = SGD_Momentum(lr, beta)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _init_cma(self):
        """CMA-ES initialization"""
        # 计算所有adapter参数的总数
        total_params = sum(p.numel() for adapter in self.model.adapters.values() 
                          for p in adapter.parameters())
        popsize = self.pertub   # which is equal to 4 + 3 * np.log(dim)
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(total_params * [0], 1, inopts=cma_opts)
        #初始候选扰动为0，初始分布方差为1
        self.popsize = es.popsize
        return es

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean

    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat

    def _apply_perturbation(self, perturbation, sign=1):
        """应用扰动到所有adapter参数"""
        start_idx = 0
        
        # 遍历所有adapter的参数
        for layer_idx in self.model.adapter_layers:
            adapter = self.model.adapters[f'adapter_{layer_idx}']
            for name, param in adapter.named_parameters():
                num_params = param.numel()
                param_perturbation = perturbation[start_idx:start_idx + num_params]
                if not isinstance(param_perturbation, torch.Tensor):
                    param_perturbation = torch.tensor(
                        param_perturbation, dtype=torch.float, device=param.device
                    )
                param.data += sign * self.zo_eps * param_perturbation.reshape(param.shape)
                start_idx += num_params

    def _save_current_adapter(self):
        """保存当前所有adapter参数的深拷贝"""
        return {
            f'adapter_{layer}': {
                k: v.clone() for k, v in self.model.adapters[f'adapter_{layer}'].state_dict().items()
            } for layer in self.model.adapter_layers
        }

    def _load_adapter(self, adapter_states):
        """加载所有adapter参数"""
        for layer in self.model.adapter_layers:
            self.model.adapters[f'adapter_{layer}'].load_state_dict(
                adapter_states[f'adapter_{layer}']
            )

    def _get_perturbations(self):
        """根据不同模式生成扰动"""
        if self.mode == 'full':
            # 完整COZO: N(m,C)
            perturbations = torch.tensor(np.array(self.es.ask()), device='cuda')
            
        elif self.mode == 'mean_only':
            # 只使用均值: m + N(0,1)
            perturbations = torch.tensor(np.array(self.es.ask()), device='cuda')  # 先调用ask()更新状态
            mean = self.es.mean
            noise = torch.randn(self.pertub, len(mean), device='cuda')
            perturbations = torch.tensor(mean, device='cuda') + noise
            
        elif self.mode == 'cov_only':
            # 只使用协方差: N(0,C)
            perturbations = torch.tensor(np.array(self.es.ask()), device='cuda')  # 先调用ask()更新状态
            B, D = self.es.B, self.es.D
            z = torch.randn(self.pertub, len(D), device='cuda')
            perturbations = torch.tensor(np.dot(z.cpu(), (B * D).T), device='cuda')
            
        # 添加零扰动
        perturbations = torch.cat([perturbations, torch.zeros_like(perturbations[0]).unsqueeze(0)])
        return perturbations

    def forward(self, x):
        """使用CMA-ES优化adapter参数"""
        # 获取用于计算移位方向的移位向量。
        shift_vector = self._get_shift_vector()

        # 初始化变量以跟踪最佳损失、对应的输出和批量均值。
        self.best_loss, self.best_outputs, batch_means = np.inf, None, []

        # 保存当前模型里adapter参数
        current_adapter = self._save_current_adapter()
        perturbations = self._get_perturbations()#生成popsize个子代
        losses = []

        # 计算梯度估计 - 修改为PyTorch tensor
        grad_estimate = torch.zeros(perturbations[0].shape, dtype=torch.float, device='cuda')
        
        # 先计算当前参数位置的损失
        self._load_adapter(current_adapter)  # 重置到初始状态
        outputs_curr, loss_curr, batch_mean_curr = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask
        )
        batch_means.append(batch_mean_curr[-768:].unsqueeze(0))
        
        for j, z in enumerate(perturbations):
            # 正向扰动
            self._load_adapter(current_adapter)  # 重置到初始状态
            self._apply_perturbation(z/self.perturbation_scale, sign=1)  #应用扰动到adapter参数
            
            outputs_pos, loss_pos, batch_mean = forward_and_get_loss(
                x, self.model, self.fitness_lambda, self.train_info, 
                shift_vector, self.imagenet_mask
            )
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean

            if self.best_loss > loss_pos.item():
                self.best_adapter = self._save_current_adapter()
                self.best_loss = loss_pos.item()
                self.best_outputs = outputs_pos
                outputs_pos = None
            losses.append(loss_pos.item())
            del outputs_pos
            
            # 计算该扰动的梯度估计 - 直接使用GPU tensor
            grad_estimate += z/self.perturbation_scale * (loss_pos - loss_curr) / (1.0 * self.zo_eps)
            print(f'Solution:[{j+1}/{len(perturbations)}], Loss: {loss_pos.item()}')
        
        # 平均梯度估计
        grad_estimate = grad_estimate / len(perturbations)
        
        # 使用优化器更新参数
        update = self.optimizer.step(grad_estimate)
        self._load_adapter(current_adapter)  # 重置到初始状态
        self._apply_perturbation(update, sign=1)  # 使用优化器计算的更新量
        
        # 计算更新后的输出和损失
        final_outputs, self.final_loss, batch_mean = forward_and_get_loss(
            x, self.model, self.fitness_lambda, self.train_info,
            shift_vector, self.imagenet_mask
        )
        
        # 更新CMA-ES状态
        self.es.tell(perturbations.cpu().numpy(), losses)
        
        # 更新历史统计信息
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_mean[-768:])
        
        return final_outputs
    
    def obtain_origin_stat(self, train_loader):
        """
        计算训练集特征的均值和方差，支持保存和加载计算结果。

        该函数用于计算训练集（源域）特征的均值和方差，为模型量化做准备。
        该函数首先尝试从保存的文件加载预计算的统计信息。
        如果找不到保存的数据，则重新计算并保存结果。
        先遍历训练集以提取特征，然后计算这些特征的均值和方差。
        最后，它为快速适应准备量化模型。

        参数:
        - train_loader: DataLoader 类型，训练集的数据加载器。

        返回:
        无
        """
        print('===> Start calculating mean and variance')
        # 创建保存目录
        save_dir = os.path.join('dataset', 'train_stats')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'train_info_adapter.pt')
        
        # 尝试加载已保存的统计信息
        if os.path.exists(save_path):
            print('===> 从文件加载预计算的均值和方差')
            saved_data = torch.load(save_path)
            self.train_info = saved_data['train_info']
        else:
            print('===> 开始计算均值和方差')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()               #dl[0]是图片，dl[1]是标签
                    feature = self.model.layers_cls_features(images)    #从ViT模型中提取图像特征
                    features.append(feature)
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)#计算源域数据特征的均值和方差，dim=0表示计算每个特征的均值和方差
            del features#释放内存
            
            # 保存计算结果
            print('===> 保存计算结果到文件')
            torch.save({
                'train_info': self.train_info,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            }, save_path)
        
        # 为快速适应准备量化模型
        for _, m in self.model.vit.named_modules():  # 遍历ViT模型的所有模块
            if type(m) == PTQSLBatchingQuantMatMul:  # 如果是PTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(
                    torch.zeros((1,12,197,64)).cuda(),  # 移除prompt tokens
                    torch.zeros((1,12,64,197)).cuda()   # 移除prompt tokens
                )
            elif type(m) == SoSPTQSLBatchingQuantMatMul:  # 如果是SoSPTQSLBatchingQuantMatMul类型
                m._get_padding_parameters(
                    torch.zeros((1,12,197,197)).cuda(),  # 移除prompt tokens
                    torch.zeros((1,12,197,64)).cuda()    # 移除prompt tokens
                )
        print('===> 计算均值和方差结束')

    def reset(self):
        """重置优化器和模型状态"""
        self.es = self._init_cma()
        self.hist_stat = None
        self.model.reset_adapters()
        self.best_adapter = {k: v.state_dict() for k, v in self.model.adapters.items()}
        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()

    def monitor_covariance(self):
        """监控协方差矩阵的状态"""
        condition_number = np.max(self.es.D) / np.min(self.es.D)
        eigenvalue_distribution = self.es.D
        return {
            'condition_number': condition_number,
            'eigenvalue_distribution': eigenvalue_distribution,
            'mean': self.es.mean if self.mode != 'cov_only' else None
        }

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """计算softmax分布的熵"""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model: AdaFormerViT, fitness_lambda, train_info, shift_vector, imagenet_mask):
    """前向传播并计算损失"""
    features = model.layers_cls_features_with_adapters(images)#提取12层cls特征，变成(64,12*768)=(64,9216)
    cls_features = features[:, -768:] # the feature of classification token，ViT模型的最后一个分类token的特征768维，e_N^0
    
    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0) #OOD域数据特征的均值和方差
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])#ID和OOD域数据特征MSE损失
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    output = model.vit.head(cls_features)#通过分类头，将cls_features映射到分类输出
    
    # 添加调试信息
    # print(f"Raw output stats - min: {output.min():.6f}, max: {output.max():.6f}, mean: {output.mean():.6f}")
    # print(f"Output shape: {output.shape}")
    
    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]#有需要可以利用imagenet_mask对输出进行筛选
    entropy_loss = softmax_entropy(output).sum()#对batch求和得到熵损失
    loss = discrepancy_loss + entropy_loss
    
    # 添加调试信息
    # print(f"Discrepancy loss: {discrepancy_loss:.6f}, Entropy loss: {entropy_loss:.6f}, Total loss: {loss:.6f}")
    
    """activation shifting, Eqn. (7)"""
    if shift_vector is not None:
        output = model.vit.head(cls_features + 1. * shift_vector)#添加activate shift 偏移量，覆盖原有output
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

    return output, loss, batch_mean