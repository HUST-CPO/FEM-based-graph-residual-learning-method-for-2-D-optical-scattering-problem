import os
import argparse

# 【关键】必须在导入 torch 之前设置 CUDA_VISIBLE_DEVICES
# 否则环境变量不会生效，torch 会使用所有可见的 GPU

# 从全局配置导入默认设置
try:
    from config import DEFAULT_TARGET_GPUS
except ImportError:
    # 如果config.py不存在，使用默认值
    DEFAULT_TARGET_GPUS = "2,3"

# 如果环境变量未设置，先设置默认值
# 这样即使作为模块导入，也能正确设置
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_TARGET_GPUS
    print(f"🔧 设置默认 CUDA_VISIBLE_DEVICES={DEFAULT_TARGET_GPUS} (在导入 torch 之前)")

# 设置NCCL环境变量以避免超时
# NCCL超时时间设置为60分钟（3600000毫秒），增加超时时间以避免心跳超时
os.environ["NCCL_TIMEOUT"] = "3600000"
os.environ["NCCL_IB_TIMEOUT"] = "3600000"
# 设置NCCL心跳超时时间（默认30秒，增加到5分钟）
os.environ["NCCL_HEARTBEAT_TIMEOUT_SEC"] = "300"
# 启用NCCL调试（可选，用于诊断问题）
# os.environ["NCCL_DEBUG"] = "INFO"
# 禁用NCCL的异步错误处理，避免进程被意外终止
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
print(f"🔧 设置NCCL超时时间为60分钟，心跳超时为5分钟以避免通信超时")

# 现在可以安全地导入 torch
import torch

# 导入训练主函数
# 支持两种训练模式：DataParallel (train.py) 和 DDP (train-1.py)
try:
    from train import main as dp_main  # DataParallel 版本
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

try:
    from trainddp import main as ddp_main  # DDP 版本
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

def single_gpu_train(target_gpu):
    """
    单卡训练模式（DataParallel 模式）
    """
    # 注意：这里设置已经太晚了，torch 已经初始化
    # 单卡模式需要重新启动进程才能生效
    print(f"⚠️  警告：单卡模式需要在启动脚本时设置 CUDA_VISIBLE_DEVICES")
    print(f"   例如：CUDA_VISIBLE_DEVICES={target_gpu} python run.py --mode single --gpu {target_gpu}")
    
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 GPU。")
        return
    
    print(f"✅ [Single] 单卡训练模式启动 (GPU {target_gpu})")
    print("   使用 DataParallel 模式")
    print(f"   当前可见 GPU 数量: {torch.cuda.device_count()}")
    
    try:
        dp_main()
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        raise e
    
    print("\n✅ 单卡训练结束。")

def multi_gpu_train(target_gpus):
    """
    多卡 DataParallel 训练模式
    注意：DP 模式不需要多进程，train.py 会自动使用所有可见的 GPU
    """
    if not DP_AVAILABLE:
        print("❌ 错误：未找到 train.py (DataParallel 版本)")
        print("   请确保 train.py 文件存在")
        return

    # 检查环境变量
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if current_visible != target_gpus:
        print(f"⚠️  警告: CUDA_VISIBLE_DEVICES={current_visible}")
        print(f"   期望设置为: {target_gpus}")
        print(f"   环境变量在导入 torch 之前已设置，当前值可能来自默认配置或之前的设置")
        print(f"   如果这不是你想要的，请在运行脚本前设置环境变量：")
        print(f"   CUDA_VISIBLE_DEVICES={target_gpus} python run.py --gpus {target_gpus}")

    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 GPU。")
        return

    # 计算实际可用卡数
    n_gpus = torch.cuda.device_count()
    expected_gpus = len(target_gpus.split(','))

    if n_gpus != expected_gpus:
        print(f"⚠️  警告：配置了 {expected_gpus} 张卡 ({target_gpus})，但系统仅检测到 {n_gpus} 张。")
        if n_gpus == 0:
            return

    print(f"✅ [Multi-DP] 检测到 {n_gpus} 张可用显卡")
    print(f"   使用 DataParallel 模式进行多卡训练")
    print(f"   配置的物理 GPU: {target_gpus}")
    print(f"   当前 CUDA_VISIBLE_DEVICES: {current_visible}")
    print("----------------------------------------------------------------")

    try:
        # DP 模式下直接调用 main()，train.py 会自动使用 DataParallel 包装模型
        dp_main()
    except Exception as e:
        print(f"\n❌ DataParallel 训练发生异常: {e}")
        raise e

def ddp_train(target_gpus):
    """
    多卡 DDP 分布式训练模式
    注意：DDP 模式使用多进程，每个进程负责一个 GPU
    """
    if not DDP_AVAILABLE:
        print("❌ 错误：未找到 trainddp.py (DDP 版本)")
        print("   请确保 trainddp.py 文件存在")
        return

    # 检查环境变量
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if current_visible != target_gpus:
        print(f"⚠️  警告: CUDA_VISIBLE_DEVICES={current_visible}")
        print(f"   期望设置为: {target_gpus}")
        print(f"   环境变量在导入 torch 之前已设置，当前值可能来自默认配置或之前的设置")
        print(f"   如果这不是你想要的，请在运行脚本前设置环境变量：")
        print(f"   CUDA_VISIBLE_DEVICES={target_gpus} python run.py --gpus {target_gpus}")

    if not torch.cuda.is_available():
        print("❌ 错误：未检测到 GPU。")
        return

    # 计算实际可用卡数
    n_gpus = torch.cuda.device_count()
    expected_gpus = len(target_gpus.split(','))

    if n_gpus != expected_gpus:
        print(f"⚠️  警告：配置了 {expected_gpus} 张卡 ({target_gpus})，但系统仅检测到 {n_gpus} 张。")
        if n_gpus == 0:
            return

    print(f"✅ [Multi-DDP] 检测到 {n_gpus} 张可用显卡")
    print(f"   使用 DDP 分布式训练模式")
    print(f"   配置的物理 GPU: {target_gpus}")
    print(f"   当前 CUDA_VISIBLE_DEVICES: {current_visible}")
    print("----------------------------------------------------------------")

    try:
        # DDP 模式下调用 ddp_main()，它会自动启动多进程
        ddp_main()
    except Exception as e:
        print(f"\n❌ DDP 训练发生异常: {e}")
        raise e

def main_launcher():
    """
    启动器主函数
    支持两种方式指定 GPU：
    1. 通过命令行参数（优先级更高）
    2. 通过代码中的配置区域（默认值）
    
    注意：CUDA_VISIBLE_DEVICES 必须在导入 torch 之前设置
    如果通过命令行参数指定了不同的 GPU，需要重新启动脚本
    """
    parser = argparse.ArgumentParser(description='PhiSAGE 训练启动器')
    parser.add_argument('--mode', type=str, choices=['single', 'multi', 'ddp'],
                       help='训练模式: single (单卡), multi (多卡DataParallel), ddp (多卡DDP)')
    parser.add_argument('--gpu', type=str, 
                       help='指定 GPU: 单卡模式传入单个数字 (如 "0")，多卡模式传入逗号分隔的列表 (如 "0,1,2" 或 "4,5,6")')
    parser.add_argument('--gpus', type=str, 
                       help='多卡模式的 GPU 列表 (与 --gpu 功能相同，用于多卡模式更清晰)')
    
    args = parser.parse_args()
    
    # ================= 配置区域（从全局配置导入）=================
    # 从全局配置导入默认值，命令行参数可以覆盖这些值

    # 导入全局配置
    try:
        from config import DEFAULT_TRAIN_MODE, DEFAULT_SINGLE_GPU_ID
        train_mode = args.mode if args.mode else DEFAULT_TRAIN_MODE
        single_gpu_id = DEFAULT_SINGLE_GPU_ID
    except ImportError:
        # 如果config.py不存在，使用本地默认值
        train_mode = args.mode if args.mode else "ddp"
        single_gpu_id = 0

    # 多卡配置 (从全局配置导入)
    target_gpus = DEFAULT_TARGET_GPUS
    
    # ===========================================
    
    # 命令行参数覆盖默认配置
    if args.gpu:
        if train_mode == "single":
            try:
                single_gpu_id = int(args.gpu)
            except ValueError:
                print(f"❌ 错误：单卡模式下 --gpu 必须是单个数字，当前值: {args.gpu}")
                return
        else:  # multi mode
            target_gpus = args.gpu
    
    if args.gpus:
        if train_mode == "multi":
            target_gpus = args.gpus
        else:
            print("⚠️  警告：--gpus 参数仅在 multi 模式下有效，当前模式为 single，忽略此参数")
    
    # 检查环境变量是否匹配
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if train_mode == "multi" and current_visible != target_gpus:
        print("=" * 60)
        print("⚠️  重要提示")
        print("=" * 60)
        print(f"   命令行指定了 GPU: {target_gpus}")
        print(f"   但 CUDA_VISIBLE_DEVICES 已设置为: {current_visible}")
        print(f"   环境变量在导入 torch 之前已设置，无法更改")
        print(f"   要使用指定的 GPU，请重新运行：")
        print(f"   CUDA_VISIBLE_DEVICES={target_gpus} python run.py --gpus {target_gpus}")
        print("=" * 60)
        print()
        # 继续运行，但使用当前的环境变量设置
    
    # 打印配置信息
    print("=" * 60)
    print("📋 训练配置")
    print("=" * 60)
    mode_desc = {
        "single": "单卡训练",
        "multi": "多卡 DataParallel",
        "ddp": "多卡 DDP 分布式训练"
    }
    print(f"   模式: {train_mode} ({mode_desc.get(train_mode, '未知')})")
    if train_mode == "single":
        print(f"   使用 GPU: {single_gpu_id}")
    else:
        print(f"   使用 GPU: {target_gpus}")
    print(f"   当前 CUDA_VISIBLE_DEVICES: {current_visible}")

    # 显示可用模式
    available_modes = []
    if DP_AVAILABLE or DDP_AVAILABLE:
        if DP_AVAILABLE:
            available_modes.append("DataParallel (train.py)")
        if DDP_AVAILABLE:
            available_modes.append("DDP (trainddp.py)")
    print(f"   可用训练后端: {', '.join(available_modes)}")
    print("=" * 60)
    print()

    if train_mode == "single":
        single_gpu_train(single_gpu_id)
    elif train_mode == "multi":
        multi_gpu_train(target_gpus)
    elif train_mode == "ddp":
        ddp_train(target_gpus)
    else:
        print("❌ 未知模式，请检查 train_mode 设置")
        print("   可选值: 'single', 'multi', 或 'ddp'")

if __name__ == "__main__":
    # 限制 CPU 线程数，避免 CPU 争抢
    # 注意：DP 模式是单进程多线程，不需要像 DDP 那样严格限制
    # 但设置这些环境变量仍然有助于避免资源争抢
    os.environ["OMP_NUM_THREADS"] = "4"  # DP 模式下可以适当增加
    os.environ["MKL_NUM_THREADS"] = "4"
    
    main_launcher()
