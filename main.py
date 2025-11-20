import os
import time
import datetime
import argparse
import random
import math
from importlib import reload, import_module

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask

import torch    
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter# 导入 TensorBoard
import timm
import numpy as np

import tta_library.tent as tent
import tta_library.sar as sar
import tta_library.cotta as cotta
import tta_library.foa_bp as foa_bp
import tta_library.rotta as rotta
import tta_library.deyo as deyo
import tta_library.eata as eata

from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.foa import FOA
from tta_library.foa_shift import Shift
from tta_library.lame import LAME

from calibration_library.metrics import ECELoss

from quant_library.quant_utils.models import get_net
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

from models.vpt import PromptViT
from models.vit_adapter import AdapterViT
from models.adaformer import AdaFormerViT




def validate_adapt(val_loader, model, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        total_batches = len(val_loader)
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            
            # 前向传播
            output = model(images)
            
            # 获取当前损失值（从不同算法模型中）
            current_loss = None
            if hasattr(model, 'final_loss'):  # 对于CMA_ZO, FOA, COZO, Tent, SAR, CoTTA等模型
                current_loss = model.final_loss
                #虽然添加了Tent, SAR, CoTTA的final_loss，但是这些方法loss函数定义不同，无法进行一起比较（注意）

            # for calculating Expected Calibration Error (ECE)
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            # 记录到tensorboard，使用corruption类型作为区分
            if i % 5 == 0:  # 每5个batch记录一次
                # 使用corruption类型和severity作为tag前缀
                prefix = f"{args.corruption}"
                if hasattr(args, 'severity'):
                    prefix = f"{prefix}/severity_{args.severity}"
                
                # 记录准确率
                writer.add_scalar(f'Accuracy/Top1/{prefix}', top1.avg, i)
                writer.add_scalar(f'Accuracy/Top5/{prefix}', top5.avg, i)
                
                # 记录损失值（如果有）
                if current_loss is not None:
                    writer.add_scalar(f'Loss/{prefix}', current_loss, i)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Calculate and display remaining time every 5 batches
            if i % 5 == 0:
                remaining_time = batch_time.avg * (total_batches - i)
                remaining_time_str = str(datetime.timedelta(seconds=int(remaining_time)))
                print(f"Remaining time: {remaining_time_str} ")
                logger.info(progress.display(i))

        # 计算ECE
        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()
        
        logits = args.algorithm != 'lame' # only lame outputs probability
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE
        # 记录最终的评估指标
        writer.add_scalar(f'Final/{args.corruption}/Top1', top1.avg, 0)
        writer.add_scalar(f'Final/{args.corruption}/Top5', top5.avg, 0)
        writer.add_scalar(f'Final/{args.corruption}/ECE', ece_avg, 0)
    return top1.avg, top5.avg, ece_avg

def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    train_dataset.switch_mode(True, False)
    return train_dataset, train_loader

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./quant_library/configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"quant_library.configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_v2', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_sketch', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_adv', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/dockerdata/imagenet-r', help='path to corruption dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')#服务器的 CPU 核数为 72，主机为 12，workers 一般设为 2/3，即 48
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # algorithm selection
    parser.add_argument('--algorithm', default='foa', type=str, help='supporting foa, sar, cotta and etc.')
    parser.add_argument('--continue_learning', default=False, type=bool, help='whether to use continue learing or reset.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--dataset_style', default='imagenet_c', type=str, help='dataset style: imagenet_c, imagenet_r_s_v2, other')

    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_base', 'resnet50'], help='model architecture: vit_base or resnet50')

    # foa settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA-BP')
    
    #foa_zo added
    parser.add_argument('--num_samples', default=20, type=int, help='number of random samples for Zero-order gradient estimation.')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Zero-order perturbation magnitude for FOA_ZO algorithm (default: 0.1)')
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for optimization (default: 0.01)")
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum coefficient for SGD-M (default: 0.9)')
    parser.add_argument('--lambda_factor', default=0.1, type=float, help='controls the update rate for covariance matrix in FOA_ZO_Covariance')
    # Adam优化器参数
    parser.add_argument('--beta1', default=0.9, type=float, help='Beta1 coefficient for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', default=0.999, type=float, help='Beta2 coefficient for Adam optimizer (default: 0.999)')
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon for Adam optimizer (default: 1e-8)')
    

    # compared method settings
    parser.add_argument('--margin_e0', default=0.4*math.log(1000), type=float, help='the entropy margin for sar')    

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')
    parser.add_argument('--root_log_dir', type=str, default='./logs', help='Root directory for logs')

    # 添加pertub参数
    parser.add_argument('--pertub', default=13, type=int, help='number of perturbations for CMA-ZO algorithm')
    parser.add_argument('--adapter_layer', default='11', type=str, 
                       help='adapter layer for CMA-ZO-Adapter algorithm. Single number (e.g., 11) or comma-separated numbers (e.g., 3,7,11)')
    parser.add_argument('--reduction_factor', default=48, type=int, help='reduction factor for CMA-ZO-Adapter algorithm')
    parser.add_argument('--adapter_style', default="parallel", type=str, help='choose the style of adaformer: parallel or sequential')
    # parser.add_argument('--perturbation_scale', default=10.0, type=float, help='perturbation scale for CMA-ZO-Adapter algorithm')

    # cozo_ablation settings
    parser.add_argument('--mode', default='full', type=str, choices=['full', 'mean_only', 'cov_only'],
                       help='mode for COZO ablation study: full, mean_only, or cov_only')

    # 添加cozo优化器相关参数
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'sgd_momentum'],
                       help='optimizer type for COZO algorithm')
    parser.add_argument('--beta', default=0.9, type=float,
                       help='momentum coefficient for SGD with momentum')

    # 添加CAZO相关参数
    parser.add_argument('--nu', default=0.8, type=float, 
                       help='Decay factor for diagonal Hessian estimation in CAZO')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 初始化 TensorBoard SummaryWriter
    log_dir = os.path.join(args.root_log_dir, f"{args.algorithm}{args.tag}")
    os.makedirs(log_dir, exist_ok=True)  # 创建目录（如果不存在）
    writer = SummaryWriter(log_dir=log_dir)  # 你可以自定义日志路径和实验名称

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    
    # 记录GPU设备信息
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    logger.info(f"Using GPU: {args.gpu}")
    
    logger.info(args)

    # configure the domains for adaptation
    # options for ImageNet-R/V2/Sketch are ['rendition', 'v2', 'sketch']
    # For ImageNet-R, the fitness_lambda of FOA should be set to 0.2
    # We advise parallelizing the experiments for FOA (K=28) on multiple GPUs, where each GPU only run a corruption
    if args.dataset_style == 'imagenet_c':
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.dataset_style == 'imagenet_r_s_v2':
        corruptions = ['rendition', 'v2', 'sketch']
    elif args.dataset_style == 'other':
        corruptions = ['gaussian_noise']
    else:
        raise NotImplementedError("Only imagenet_c and imagenet_r_s_v2 are supported for now.")

    # create model
    if args.quant:
        # Use PTQ4Vit for model quantization
        # NOTE the bit of quantization can be modified in quant_library/configs/PTQ4ViT.py
        quant_cfg = init_config("PTQ4ViT")
        net = get_net('vit_base_patch16_224')
        wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)   # 将模型中的模块包装成量化模块
        g=datasets.ViTImageNetLoaderGenerator(args.data,'imagenet',32,32,16,kwargs={"model":net})
        test_loader=g.test_loader()
        calib_loader=g.calib_loader(num=32)  # 校准数据：32个样本
        
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
    else:
        # full precision model
        if args.arch == 'vit_base':
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif args.arch == 'resnet50':
            from models.resnet import resnet50
            net = resnet50(pretrained=True)
        else:
            raise ValueError(f'Unknown architecture: {args.arch}')
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)



    if args.algorithm == 'cozo':
        from tta_library.COZO_Ablation import COZO_Ablation  # 添加COZO_Ablation导入
        # 使用AdaFormerViT包装原始ViT模型
        net = AdaFormerViT(net, adapter_layer=args.adapter_layer, 
                           reduction_factor=args.reduction_factor, adapter_style=args.adapter_style).cuda()
        adapt_model = COZO_Ablation(
            model=net,
            mode="cov_only",
            fitness_lambda=args.fitness_lambda,
            lr=args.lr,
            pertub=args.pertub,
            epsilon=args.epsilon,
            optimizer_type=args.optimizer,
            beta=args.beta
        )
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)

    elif args.algorithm == 'cazo':
        from tta_library.CAZO import CAZO
        net = AdaFormerViT(net, adapter_layer=args.adapter_layer, 
                          reduction_factor=args.reduction_factor, adapter_style=args.adapter_style).cuda()
        adapt_model = CAZO(
            model=net,
            fitness_lambda=args.fitness_lambda,
            lr=args.lr,
            pertub=args.pertub,
            epsilon=args.epsilon,
            optimizer_type=args.optimizer,
            beta=args.beta,
            nu=args.nu,
        )
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'cazo_lit':
        from tta_library.CAZO_lit import CAZO_Lit
        net = AdaFormerViT(net, adapter_layer=args.adapter_layer, 
                          reduction_factor=args.reduction_factor, adapter_style=args.adapter_style).cuda()
        adapt_model = CAZO_Lit(
            model=net,
            fitness_lambda=args.fitness_lambda,
            lr=args.lr,
            pertub=args.pertub,
            epsilon=args.epsilon,
            optimizer_type=args.optimizer,
            beta=args.beta
        )
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'zo_base':
        from tta_library.zo_base import ZO_Base  # 导入ZO_Base算法
        # 使用ZO_Base算法
        net = AdaFormerViT(net, adapter_layer=args.adapter_layer, 
                           reduction_factor=args.reduction_factor, adapter_style=args.adapter_style).cuda()
        adapt_model = ZO_Base(
            model=net,
            fitness_lambda=args.fitness_lambda,
            lr=args.lr,
            pertub=args.pertub,
            epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1,
            optimizer_type=args.optimizer,
            beta=args.beta
        )
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foa_shift':
        # activation shifting doesn't need to insert prompts 
        net = PromptViT(net, 0).cuda()
        adapt_model = Shift(net)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foa_bp':
        # foa_bp updates the normalization layers, thus no prompt is needed
        net = PromptViT(net, 0).cuda()
        net = foa_bp.configure_model(net).cuda()
        params, _ = foa_bp.collect_params(net)
        optimizer = torch.optim.SGD(params, 0.005, momentum=0.9)
        adapt_model = foa_bp.FOA_BP(net, optimizer, args.lambda_bp)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 't3a':
        # NOTE: set num_classes to 200 on ImageNet-R
        adapt_model = T3A(net, 1000, 20).cuda()
    elif args.algorithm == 'tent':
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, 0.001, momentum=0.9)
        adapt_model = tent.Tent(net, optimizer)
    elif args.algorithm == 'foa':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = FOA(net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foa_resnet':
        import tta_library.foa_resnet as foa_resnet
        from models.prompt_resnet import PromptResNet
        net = foa_resnet.configure_model(net).cuda()
        net = PromptResNet(args, net).cuda()
        adapt_model = foa_resnet.FOA_ResNet(args, net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'cazo_resnet':
        import tta_library.CAZO_resnet as CAZO_resnet
        from models.prompt_resnet import PromptResNet
        
        net = CAZO_resnet.configure_model(net).cuda()
        net = PromptResNet(args, net).cuda()
        adapt_model = CAZO_resnet.CAZO_ResNet(
            args=args,
            model=net,
            fitness_lambda=args.fitness_lambda,
            lr=args.lr,
            pertub=args.pertub,
            epsilon=args.epsilon,
            optimizer_type=args.optimizer,
            beta=args.beta,
            nu=args.nu
        )
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
        # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.margin_e0)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    elif args.algorithm == 'lame':
        adapt_model = LAME(net)
    elif args.algorithm == 'rotta':
        net = rotta.configure_model(net)
        params, _ = rotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0)
        adapt_model = rotta.RoTTA(net, optimizer, nu=0.001, memory_size=64, 
                                   update_frequency=64, steps=1, episodic=False)
    elif args.algorithm == 'deyo':
        net = deyo.configure_model(net)
        params, _ = deyo.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = deyo.DeYO(net, optimizer, num_classes=1000, 
                                 reweight_ent=True, reweight_plpd=True,
                                 plpd_threshold=0.2, margin=0.5, margin_e0=0.4,
                                 aug_type='pixel', steps=1, episodic=False)
    elif args.algorithm == 'eata':
        net = eata.configure_model(net)
        params, _ = eata.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, num_classes=1000,
                                 margin_e0=0.4, d_margin=0.05, 
                                 fisher_alpha=2000.0, fishers=None,
                                 steps=1, episodic=False)
    elif args.algorithm == 'eta':
        net = eata.configure_model(net)
        params, _ = eata.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = eata.ETA(net, optimizer, num_classes=1000,
                                margin_e0=0.4, d_margin=0.05,
                                steps=1, episodic=False)
    elif args.algorithm == 'no_adapt':
        adapt_model = net#直接validate_adapt
    else:
        assert False, NotImplementedError


    corrupt_acc, corrupt_ece = [], []
    for corrupt in corruptions:
        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        else:
            adapt_model.imagenet_mask = None

        val_dataset, val_loader = prepare_test_data(args)

        torch.cuda.empty_cache()
        top1, top5, ece_loss = validate_adapt(val_loader, adapt_model, args, writer)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}")
        corrupt_acc.append(top1)
        corrupt_ece.append(ece_loss)
        
        logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
        logger.info(f'mean ece of corruption: {sum(corrupt_ece)/len(corrupt_ece)*100 if len(corrupt_ece) else 0}')
        logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
        logger.info(f'corrupt ece list: {[_*100 for _ in corrupt_ece]}')

        # reset model before adapting on the next domain
        if args.algorithm == 'no_adapt':
            continue
        elif args.continue_learning:
            continue
        else:
            print("Resetting model to original weights...")
            adapt_model.reset()
    
    # 结束训练时关闭 TensorBoard writer
    writer.close()  