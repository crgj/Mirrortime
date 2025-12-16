#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import cv2
import time
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
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

# ==============================================================================
# [PUP 3D-GS Integration] Import Fisher computation module
# ==============================================================================
from fisher_pool_xyz_scaling import pool_fisher_cuda
try:
    from fisher_pool_xyz_scaling import pool_fisher_cuda
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False
    print("\n[WARNING] 'fisher_pool_xyz_scaling' not found. PUP Pruning will be skipped.\n")

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):

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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    web_logger_server.start_server()
    web_logger_server.init_logger(vars(opt), opt.iterations)
    web_logger_server.set_max_frames(scene.frame_count)


    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    first_iter += 1
    
    # [Dataset Loading]
    # Unlike standard 3DGS which often preloads all cameras or loads them one-by-one, 
    # we use a PyTorch DataLoader to efficiently handle large 4D datasets (video sequences).
    # This allows for parallel data loading and better memory management.
    torch.cuda.empty_cache()
    dataloader = DataLoader(scene.getTrainDataset(), batch_size=opt.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn, persistent_workers=True, prefetch_factor=2)
    loader_iter = iter(dataloader)
    batch_viewpoint_stack = []
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # [PUP 3D-GS Integration] Initialize pruning counter
    prune_idx = 0
    
    # WDD [2024-07-31] [For GUI Dynamic Playback]
    last_time_update = time.time()
    current_time_idx = 0
    # Use total_frames because opacity is now [N, 1] + Network
    frame_count = scene.frame_count

    for iteration in range(first_iter, opt.iterations + 1):
        # Get next batch
        if not batch_viewpoint_stack:
            try:
                batch_cameras = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                batch_cameras = next(loader_iter)
            
            # Load data to GPU
            current_batch_cameras = batch_cameras
            for cam in current_batch_cameras:
                cam.to_device("cuda")
            
            # Use this batch for opt.batch_iterations (resampling from it)
            # Actually, standard DataLoader usage implies we train on this batch once?
            # User's previous plan: "Train on this batch for multiple iterations (enough to amortize the transfer cost)"
            
            batch_viewpoint_stack = []
            # We want to train on this *set* of cameras for multiple steps.
            # So we just keep them in current_batch_cameras.
            # But the loop is 'for iteration in range...'.
            # We need to detect when to switch batch.

        # Logic:
        # We need a batch present. We hold it for `opt.batch_iterations`.
        # So we only fetch new batch if (iteration % batch_iterations == 0) etc.
        # But we are inside a big loop.

        # Let's align with previous logic:
        # If we need to switch batch (every N iterations):
        # [Dataset Loading]
        # To amortize the cost of transferring data to the GPU, we reuse the same batch of cameras 
        # for `opt.batch_iterations`. This is a deviation from standard 3DGS where the stochastic 
        # selection usually happens from the entire pool every iteration.
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


        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # WDD [2024-07-31] [Auto-cycle time index every 0.2s]
                    if time.time() - last_time_update > 0.2:
                        last_time_update = time.time()

                        current_time_idx = (current_time_idx + 1) % frame_count
                    
                    # [Lifetime/4D]
                    # Assign the current time index to the camera. This is crucial for 4D rendering
                    # as it determines which Gaussians are active/visible based on their lifetime parameters.
                    custom_cam.time_idx = current_time_idx
                    
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(f"GUI error: {e}")
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera from current batch
        if not batch_viewpoint_stack:
            batch_viewpoint_stack = current_batch_cameras.copy()
        rand_idx = randint(0, len(batch_viewpoint_stack) - 1)
        viewpoint_cam = batch_viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx) # Unused


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and render_pkg["depth"] is not None:
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

        # Web Logger Render
        render_req = web_logger_server.get_render_request()
        if render_req:
            try:
                # Expects a dict with camera params
                w = render_req.get("width", 800)
                h = render_req.get("height", 600)
                fovy = render_req.get("fovy", 1.0)
                fovx = render_req.get("fovx", 1.0)
                znear = render_req.get("znear", 0.01)
                zfar = render_req.get("zfar", 100.0)
                
                view_matrix = torch.tensor(render_req["view_matrix"]).cuda().float()
                proj_matrix = torch.tensor(render_req["proj_matrix"]).cuda().float()
                
                # Compute full proj (World2Clip) = View * Proj (in row-major / GL style it depends)
                # MiniCam expects world_view_transform (World2View) and full_proj_transform
                # Assuming incoming matrices are GL style (column major?) or row major?
                # Usually WebGL sends column-major matrices.
                # PyTorch3D / GS codebase usually uses row-major for storage but multiplies correctly.
                # Let's assume the client sends what we need or consistent matrices.
                # If render_req sends 'view_matrix' as World2View.
                
                full_proj_transform = view_matrix.unsqueeze(0).bmm(proj_matrix.unsqueeze(0)).squeeze(0)
                
                custom_cam = MiniCam(w, h, fovy, fovx, znear, zfar, view_matrix, full_proj_transform)
                
                # Render
                render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifier=1.0)
                img = render_pkg["render"]
                
                # Convert to bytes
                img_8 = (torch.clamp(img, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                img_bytes = cv2.imencode('.jpg', img_8)[1].tobytes()
                
                web_logger_server.submit_render_result(img_bytes)
            except Exception as e:
                print(f"Web Viewer Render Error: {e}")


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
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
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
                    densification_stats = gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                else:
                    densification_stats = None
                
                                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            
            if iteration % 200 == 0:
                # [Lifetime Statistics]
                # Log the distribution of 'active duration' (how long Gaussians remain visible).
                # This metric is specific to 4DGS and helps analyze temporal consistency.
                active_duration_tensor = gaussians.compute_active_duration(threshold=0.001)
                opacity_tensor = gaussians.get_opacity # Get current base opacity or combined if valid
                web_logger_server.log_metrics(iteration, loss.item(), gaussians.get_xyz.shape[0], lifetime_tensor=active_duration_tensor, densification_stats=densification_stats, opacity_tensor=opacity_tensor)
            else:
                web_logger_server.log_metrics(iteration, loss.item(), gaussians.get_xyz.shape[0], densification_stats=densification_stats)


            # ==================================================================
            # [PUP 3D-GS Integration] Pruning Logic
            # ==================================================================
            if FISHER_AVAILABLE and args.prune_iterations and iteration in args.prune_iterations:
                if prune_idx < len(args.prune_percent):
                    current_prune_percent = args.prune_percent[prune_idx]
                    print(f"\n[ITER {iteration}] Executing PUP Pruning Round {prune_idx+1}")
                    print(f"Target Pruning Percentage: {current_prune_percent * 100}%")

                    # 0. Preparation: clear gradients, ensure memory
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    # 1. Calculate Fisher Matrix
                    N = gaussians.get_xyz.shape[0]
                    device = gaussians.get_xyz.device
                    
                    # [PUP Pruning]
                    # Accumulate Fisher Information Matrix.
                    # The Fisher Matrix serves as a proxy for "sensitivity" or importance of each Gaussian.
                    # High Fisher info -> removing this point causes large change in loss -> Important.
                    # Low Fisher info -> removing this point causes small change -> Prunable.
                    fishers = torch.zeros(N, 6, 6, device=device).float()

                    # Must enable gradients for backward pass
                    with torch.enable_grad():
                        train_cameras = scene.getTrainCameras()
                        # Flatten the batch_viewpoint_stack logic or iterate over all cameras?
                        # Original PUP iterates over all training cameras.
                        # scene.getTrainCameras() returns a list of all cameras if they are loaded.
                        # However, Mirrortime seems to use a DataLoader approach (scene.getTrainDataset()).
                        # We need to be careful here. scene.getTrainCameras() might return empty list if not loaded.
                        
                        # Check if scene.getTrainCameras() works as expected.
                        cameras_to_use = scene.getTrainCameras()
                        if not cameras_to_use:
                             # Fallback to dataset if cameras are not stored in scene (due to DataLoader)
                             if hasattr(scene, "getTrainDataset"):
                                 cameras_to_use = scene.getTrainDataset()
                             else:
                                 cameras_to_use = []

                        if len(cameras_to_use) == 0:
                             print("Error: No training cameras available for Fisher computation.")
                        else:
                            for view_idx, view in tqdm(enumerate(cameras_to_use), 
                                                    total=len(cameras_to_use), 
                                                    desc="Calculating Fisher Matrix"):
                                view.to_device("cuda")
                                # Call CUDA kernel
                                pool_fisher_cuda(
                                    view_idx, view, gaussians, pipe, background,
                                    fishers, args.fisher_resolution
                                )
                                view.to_device("cpu") # Move back to save memory
                                torch.cuda.empty_cache()
                    
                    # 2. Compute Sensitivity Score
                    print("Computing sensitivity scores...")
                    fishers_sv = torch.linalg.svdvals(fishers)
                    fishers_sv = torch.clamp(fishers_sv, min=1e-10) 
                    fishers_log_dets = torch.log(fishers_sv).sum(dim=1)

                    # 3. Determine Threshold and Generate Mask
                    n_prune = int(N * current_prune_percent)
                    if n_prune > 0:
                        topk = torch.topk(fishers_log_dets, k=n_prune, largest=False)
                        threshold = topk.values.max()
                        
                        # [PUP Pruning]
                        # Create a base mask for points with LOW sensitivity (Fisher score <= threshold).
                        prune_mask = fishers_log_dets <= threshold
                        
                        # [Lifetime/PUP Safeguard]
                        # We specifically want to PROTECT short-lived (dynamic) points from being pruned, 
                        # even if they have low sensitivity globally. Dynamic points might only be important 
                        # for a few frames, so their total Fisher info might be low, but they are critical 
                        # for those specific frames.
                        #
                        # Therefore, we strictly prune ONLY points that:
                        # 1. Have Low Sensitivity (prune_mask=True)
                        # 2. AND are "Long Duration" / Static (full_duration_mask=True)
                        #
                        # Effectively: Dynamic points are NEVER pruned by this logic.
                        gaussians.total_frames = scene.frame_count
                        active_durations = gaussians.get_lifetime()
                        # We define "Long Duration" as being active for at least half the video.
                        full_duration_mask = active_durations >= (scene.frame_count//2)
                        
                        # Apply the safeguard: Only prune if BOTH low sensitivity AND long duration.
                        prune_mask = torch.logical_and(prune_mask, full_duration_mask)
                        
                        n_final_prune = prune_mask.sum().item()
                        print(f"Pruning {n_final_prune} Gaussians (Initial target: {n_prune}, filtered by duration: {n_prune - n_final_prune})...")
                        gaussians.prune_points(prune_mask)
                    
                    # 5. Cleanup
                    del fishers, fishers_sv, fishers_log_dets
                    if 'topk' in locals(): del topk
                    if 'prune_mask' in locals(): del prune_mask
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    print(f"Pruning Complete. Remaining Gaussians: {gaussians.get_xyz.shape[0]}")
                    prune_idx += 1
                else:
                    print(f"\n[WARNING] Iteration {iteration} is in prune_iterations but no matching percentage found.")


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

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,20_000,30_000,50_000,80_000,100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # [PUP 3D-GS] Added arguments
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[10_000,20_000,30_000,40_000], help="Iterations to trigger pruning")
    parser.add_argument("--prune_percent", nargs="+", type=float, default=[0.9,0.7,0.5,0.5], help="Percentage of Gaussians to prune at each trigger")
    parser.add_argument("--fisher_resolution", type=int, default=1, help="Resolution scaling for Fisher computation")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
