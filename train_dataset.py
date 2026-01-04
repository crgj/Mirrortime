import os
import torch
import cv2
import time
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui,render_fastgs
from utils import web_logger_server
from scene.cameras import MiniCam
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from torch.utils.data import DataLoader
from scene.dataset import FourDDataset

def custom_collate_fn(batch):
    # Batch is a list of Camera objects
    return batch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
#FAST
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    background2 = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    web_logger_server.start_server()
    web_logger_server.init_logger(vars(opt), opt.iterations)
    web_logger_server.set_max_frames(scene.frame_count)


    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    first_iter += 1
    
    # DataLoader setup
    torch.cuda.empty_cache()
    dataloader = DataLoader(scene.getTrainDataset(), batch_size=opt.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn, persistent_workers=True, prefetch_factor=2)
    loader_iter = iter(dataloader)
    batch_viewpoint_stack = []
    current_batch_cameras=None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # For GUI Dynamic Playback
    last_time_update = time.time()
    current_time_idx = 0
    # Use total_frames because opacity is now [N, 1] + Network
    frame_count = scene.frame_count

    for iteration in range(first_iter, opt.iterations + 1):

        #SIBR查看实时渲染
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # Auto-cycle time index every 0.2s
                    if time.time() - last_time_update > 0.2:
                        last_time_update = time.time()

                        current_time_idx = (current_time_idx + 1) % frame_count
                    custom_cam.time_idx = current_time_idx
                    
                    net_image = render(custom_cam, gaussians, pipe, background2, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 批量加载逻辑：
        # 为了摊薄 CPU 到 GPU 的数据传输开销，我们每隔 opt.batch_iterations 次迭代更新一次相机批次。
        # 在这期间，优化器将持续在当前缓存的相机集合中进行采样训练。
        if (iteration - first_iter) % opt.batch_iterations == 0:
             # Release old
            if current_batch_cameras:
                 for cam in current_batch_cameras:
                     cam.release()
            
            # Fetch new
            try:
                current_batch_cameras = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                current_batch_cameras = next(loader_iter)
            
            for cam in current_batch_cameras:
                cam.to_device("cuda")
                
            batch_viewpoint_stack = current_batch_cameras.copy()

        iter_start.record()

        # Pick a random Camera from current batch
        if not batch_viewpoint_stack:
            batch_viewpoint_stack = current_batch_cameras.copy()
        rand_idx = randint(0, len(batch_viewpoint_stack) - 1)
        viewpoint_cam = batch_viewpoint_stack.pop(rand_idx)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.tensor([[0, 0, 0], [1, 1, 1], [0.18, 0.18, 0.18]], dtype=torch.float32, device="cuda")[randint(0, 2)] if opt.random_background else background
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            # image *= alpha_mask
            gt_image=gt_image*alpha_mask+(1-alpha_mask)*bg.view(3, 1, 1)
            

        
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)


        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()



        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # densification_stats = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    # 获取当前批次的相机列表副本
                    camlist = current_batch_cameras.copy()
                    
                    # 使用 FastGS 的多视图一致性指标计算重要性评分（importance_score）和剪枝评分（pruning_score）
                    # 这是为了确保稠密化和剪枝操作在多个视角下是鲁棒且一致的
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, background, opt, DENSIFY=True)

                    # # 对生命周期较短的点进行特殊保护
                    # lifetime_sum = gaussians.get_lifetime() # 计算每个高斯点在时间轴上的累计激活时长
                    # low_lifetime_mask = lifetime_sum < opt.low_lifetime_threshold # 识别活跃时间过短的点
                    
                    # # 保护逻辑：不剪枝活跃时间短的点，并强制对其进行稠密化处理
                    # pruning_score[low_lifetime_mask] = 0 # 保护：生命周期短的点不参与剪枝
                    # # 强制稠密化：调高重要性评分，使其超过阈值
                    # importance_score[low_lifetime_mask] = importance_score[low_lifetime_mask]*opt.importance_lambda+opt.importance_score_threshold-1 
                    
                    # 执行 FastGS 特有的稠密化与剪枝操作
                    densification_stats=gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)
                    
                else:
                    densification_stats = None
                
                                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # if iteration % opt.opacity_reset_interval == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations:
            #     camlist = current_batch_cameras.copy()
                    
            #     _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
            #     lifetime_sum = gaussians.get_lifetime()
            #     low_lifetime_mask = lifetime_sum < 3
            #     pruning_score[low_lifetime_mask] = 0 # Do not prune                 
            #     gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
            
            if iteration % 200 == 0:
                # Log Active Duration (histogram)
                active_duration_tensor = gaussians.get_lifetime()
                opacity_tensor = gaussians.get_opacity # Get current base opacity or combined if valid
                web_logger_server.log_metrics(iteration, loss.item(), gaussians.get_xyz.shape[0], densification_stats=densification_stats)
                # web_logger_server.log_metrics(iteration, loss.item(), gaussians.get_xyz.shape[0], lifetime_tensor=active_duration_tensor, densification_stats=densification_stats, opacity_tensor=opacity_tensor)
            else:
                web_logger_server.log_metrics(iteration, loss.item(), gaussians.get_xyz.shape[0], densification_stats=densification_stats)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cameras = scene.getTrainCameras()
        if not train_cameras:
             train_cameras = scene.getTrainDataset()
        
        test_cameras = scene.getTestCameras()
        if not test_cameras:
             test_cameras = scene.getTestDataset()

        validation_configs = ({'name': 'test', 'cameras' : test_cameras}, 
                              {'name': 'train', 'cameras' : [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.to_device("cuda")
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    viewpoint.release()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    # Automatic iteration generation if not provided
    if args.test_iterations is None:
        args.test_iterations = [7000,10000,30000]
        if args.densify_until_iter not in args.test_iterations:
            args.test_iterations.append(args.densify_until_iter)
        last = max(args.test_iterations)
        if args.iterations - last > 50000:
            for i in range(last + 50000, args.iterations, 50000):
                args.test_iterations.append(i)
        if args.iterations not in args.test_iterations:
            args.test_iterations.append(args.iterations)
        args.test_iterations = sorted(list(set(args.test_iterations)))

    if args.save_iterations is None:
        args.save_iterations = [30000,50000,70000,args.densify_until_iter]
        last = args.densify_until_iter
        if args.iterations - last > 30000:
            for i in range(last + 30000, args.iterations, 30000):
                args.save_iterations.append(i)
        if args.iterations not in args.save_iterations:
            args.save_iterations.append(args.iterations)
        args.save_iterations = sorted(list(set([x for x in args.save_iterations if x > 0])))
    elif args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    print("Test iterations: ", args.test_iterations)
    print("Save iterations: ", args.save_iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
