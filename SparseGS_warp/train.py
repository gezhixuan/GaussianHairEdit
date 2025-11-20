import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import diptest
from guidance.sd_utils import StableDiffusion
from random import randint
from utils.loss_utils import l1_loss, ssim, local_pearson_loss, pearson_depth_loss, mask_l1_loss, l1_loss_nonzero
from utils.prune_utils import calc_diff
import matplotlib.pyplot as plt
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, normalize
import time
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import getWorld2View2
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import copy

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



class LossGuard:
    """
    Keeps a history of recent losses and flags extreme outliers.

    - window:     how many recent losses to remember
    - threshold:  skip if current loss > threshold * mean_recent
    - min_count:  don't start skipping until we have at least this many samples
    """
    def __init__(self, window=100, threshold=3.0, min_count=20):
        self.window = window
        self.threshold = threshold
        self.min_count = min_count
        self.history = []
        self.last_ratio = 1.0
        self.last_mean = None

    def _truncate(self):
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def should_skip(self, loss_value: float) -> bool:
        """Return True if this loss should be skipped (too large)."""
        return False
        v = float(loss_value)

        # Until we have enough samples, just record and never skip
        if len(self.history) < self.min_count:
            self.history.append(v)
            self._truncate()
            self.last_mean = sum(self.history) / len(self.history)
            self.last_ratio = 1.0
            return False

        mean = sum(self.history) / len(self.history)
        self.last_mean = mean
        ratio = v / (mean + 1e-8)
        self.last_ratio = ratio

        # If this is a huge outlier, skip and DO NOT add it to history
        # (so permanently bad viewpoints keep being detected as outliers)
        if ratio > self.threshold:
            return True

        # Otherwise, accept and update history
        self.history.append(v)
        self._truncate()
        return False


# -------------------------------------------------------------------------
# Warping utilities: per-camera global homography + local warp field
# -------------------------------------------------------------------------

def apply_warp(image, depth, warp_field, base_grid, global_affine):
    """
    Apply global homography + local warp_field to GT image and depth.

    image:          [C,H,W] tensor on CUDA
    depth:          [H,W] or [1,H,W] tensor or None
    warp_field:     [1,2,H,W] local offsets in *normalized* coords
    base_grid:      [1,H,W,2] identity sampling grid in [-1,1]
    global_affine:  [1,3,3] homography matrix in normalized coords

    Returns:
        warped_image: [C,H,W]
        warped_depth: [H,W] or None
        valid_mask:   [H,W] float (1 where coords in [-1,1], else 0)
    """

    device = image.device
    C, H, W = image.shape

    # 1) Global homography on base_grid
    # base_grid: [1,H,W,2] -> flatten to [1,N,2]
    grid_flat = base_grid.view(1, -1, 2)                           # [1, H*W, 2]
    ones = torch.ones(1, grid_flat.shape[1], 1, device=device)     # [1, H*W, 1]
    grid_h = torch.cat([grid_flat, ones], dim=-1)                  # [1, H*W, 3]

    # global_affine: [1,3,3]
    # output homogeneous coords: [1, H*W, 3]
    global_grid_h = torch.bmm(grid_h, global_affine.transpose(1, 2))
    gx = global_grid_h[..., 0]
    gy = global_grid_h[..., 1]
    gw = global_grid_h[..., 2].clamp_min(1e-6)

    # De-homogenize
    x_norm = gx / gw
    y_norm = gy / gw
    global_grid_flat = torch.stack([x_norm, y_norm], dim=-1)       # [1, H*W, 2]
    global_grid = global_grid_flat.view(1, H, W, 2)                # [1,H,W,2]

    # 2) Local warp offsets (normalized)
    if warp_field is not None:
        dx = warp_field[:, 0]  # [1,H,W]
        dy = warp_field[:, 1]  # [1,H,W]
        offset = torch.stack((dx, dy), dim=-1)                      # [1,H,W,2]
        grid = global_grid + offset
    else:
        grid = global_grid

    # 3) Valid mask: coords inside [-1,1]
    x = grid[..., 0]
    y = grid[..., 1]
    valid_mask = (
        (x >= -1.0) & (x <= 1.0) &
        (y >= -1.0) & (y <= 1.0)
    ).float()[0]                                                   # [H,W]

    # 4) Sample image
    warped_image = F.grid_sample(
        image.unsqueeze(0),                                        # [1,C,H,W]
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).squeeze(0)                                                   # [C,H,W]

    warped_depth = None
    if depth is not None:
        if depth.ndim == 2:
            depth_in = depth.unsqueeze(0).unsqueeze(0)             # [1,1,H,W]
        elif depth.ndim == 3:
            depth_in = depth.unsqueeze(0)                          # [1,C,H,W]
        else:
            depth_in = depth                                       # assume [1,1,H,W] or similar

        warped_depth = F.grid_sample(
            depth_in,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0).squeeze(0)                                    # [H,W]

    return warped_image, warped_depth, valid_mask


def masked_l1(pred, gt, mask):
    """
    Masked L1 loss.

    pred, gt: [C,H,W] or [H,W]
    mask:     [H,W] float 0/1
    """
    if mask is None:
        return (pred - gt).abs().mean()

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    mask_exp = mask.unsqueeze(0).expand_as(pred)                   # [C,H,W]
    diff = (pred - gt).abs() * mask_exp
    denom = mask_exp.sum().clamp_min(1e-6)
    return diff.sum() / denom


def masked_ssim(pred, gt, mask):
    """
    Masked SSIM wrapper using existing ssim().

    pred, gt: [C,H,W] or [H,W], values in [0,1]
    mask:     [H,W] float 0/1
    """
    if mask is None:
        return ssim(pred, gt)

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    mask_exp = mask.unsqueeze(0).expand_as(pred)                   # [C,H,W]
    pred_m = pred * mask_exp
    gt_m = gt * mask_exp
    return ssim(pred_m, gt_m)


def warp_regularization_loss(warp_field,
                             lambda_mag=1e-3,
                             lambda_s1=1e-2,
                             lambda_s2=1e-2):
    """
    Strong regularization on local warp_field to prevent texture distortion.

    warp_field: [1,2,H,W]

    - magnitude term    (keep displacements small)
    - 1st-order smooth  (neighbor pixels move similarly)
    - 2nd-order smooth  (bending energy / Laplacian)
    """
    if warp_field is None:
        return torch.tensor(0.0, device='cuda')

    B, C, H, W = warp_field.shape

    # 1) magnitude
    L_mag = (warp_field ** 2).mean()

    # 2) first-order smoothness
    dx = warp_field[:, :, :, 1:] - warp_field[:, :, :, :-1]        # [1,2,H,W-1]
    dy = warp_field[:, :, 1:, :] - warp_field[:, :, :-1, :]        # [1,2,H-1,W]
    L_s1 = (dx ** 2).mean() + (dy ** 2).mean()

    # 3) second-order (Laplacian)
    center = warp_field[:, :, 1:-1, 1:-1]
    left   = warp_field[:, :, 1:-1, :-2]
    right  = warp_field[:, :, 1:-1,  2:]
    up     = warp_field[:, :, :-2, 1:-1]
    down   = warp_field[:, :,  2:, 1:-1]
    lap = 4 * center - (left + right + up + down)
    L_s2 = (lap ** 2).mean()

    return lambda_mag * L_mag + lambda_s1 * L_s1 + lambda_s2 * L_s2


def save_warped_gt_images(scene, model_path, iteration):
    """
    Save warped GT images (using current global affine + warp_field) for all train cameras.
    """
    out_dir = os.path.join(model_path, f"warped_gt_{iteration}")
    os.makedirs(out_dir, exist_ok=True)

    for cam in scene.getTrainCameras():
        if not hasattr(cam, "warp_field") or not hasattr(cam, "warp_base_grid") or not hasattr(cam, "global_affine"):
            continue

        with torch.no_grad():
            img = cam.original_image.to("cuda")                    # [C,H,W]
            warped_img, _, _ = apply_warp(
                img, None,
                cam.warp_field,
                cam.warp_base_grid,
                cam.global_affine
            )
            warped_img = warped_img.clamp(0.0, 1.0)

            img_np = (warped_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{cam.image_name}.png"), img_np)


def get_warp_state(scene):
    """
    Collect warp fields and global affines for all training cameras to checkpoint them.

    Returns: dict uid -> {'warp_field': tensor, 'global_affine': tensor}
    """
    state = {}
    for cam in scene.getTrainCameras():
        if hasattr(cam, "warp_field") and hasattr(cam, "global_affine"):
            state[cam.uid] = {
                'warp_field': cam.warp_field.detach().cpu(),
                'global_affine': cam.global_affine.detach().cpu()
            }
    return state


def save_debug_comparison(render_img, gt_img_raw, warped_gt_img,
                          model_path, iteration, image_name):
    """
    Save [render | original GT | warped GT] comparison for visual debugging.

    All inputs: [C,H,W], in [0,1].
    """
    out_dir = os.path.join(model_path, "debug_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    imgs = []
    for t in [render_img, gt_img_raw, warped_gt_img]:
        t = t.detach().clamp(0.0, 1.0)
        t_np = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        t_np = cv2.cvtColor(t_np, cv2.COLOR_RGB2BGR)
        imgs.append(t_np)

    # Make sure same size
    h_min = min(im.shape[0] for im in imgs)
    w_min = min(im.shape[1] for im in imgs)
    imgs = [cv2.resize(im, (w_min, h_min), interpolation=cv2.INTER_AREA) for im in imgs]

    comp = np.concatenate(imgs, axis=1)
    fname = os.path.join(out_dir, f"iter_{iteration:06d}_{image_name}.png")
    cv2.imwrite(fname, comp)


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, step, max_cameras, prune_sched):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, step=step, max_cameras=max_cameras)
    gaussians.training_setup(opt)
    
    freeze_gaussians_until = 4000  # only warp/affine optimize before this iter
    warp_lr_scaled = False   # NEW: to ensure we only scale once


    
    lambda_pos_reg = 1e-2
    restored_xyz = None
    
    # loss_guard = LossGuard(window=100, threshold=3.0, min_count=50)
    loss_guard = LossGuard(window=100, threshold=8.0, min_count=50)


    # ---------------------------------------------------------------------
    # Per-camera global affine + local warp field
    # ---------------------------------------------------------------------
    train_cameras = scene.getTrainCameras()
    warp_params = []
    global_affine_params = []

    for cam in train_cameras:
        H, W = cam.image_height, cam.image_width

        # Local warp field (normalized coordinates, starts at zero)
        warp_field = torch.zeros(1, 2, H, W, device="cuda", requires_grad=True)
        cam.warp_field = warp_field

        # Base sampling grid in normalized coords [-1,1]
        ys = torch.linspace(-1.0, 1.0, H, device="cuda")
        xs = torch.linspace(-1.0, 1.0, W, device="cuda")
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1,H,W,2]
        cam.warp_base_grid = base_grid

        # Global homography (3x3) in normalized space, initialized to identity
        affine = torch.tensor(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            device="cuda", dtype=torch.float32
        )
        affine.requires_grad_()
        cam.global_affine = affine

        warp_params.append(warp_field)
        global_affine_params.append(affine)

        # Separate optimizer with different LRs for local warp and global affine
        # warp_lr = 1e-4          # smaller LR for local warp_field (texture-preserving)
        # affine_lr = 1e-3       # larger LR for global affine (big shifts/scales allowed)
        warp_lr = 0         # smaller LR for local warp_field (texture-preserving)
        affine_lr = 1e-2       # larger LR for global affine (big shifts/scales allowed)
        warp_optimizer = torch.optim.Adam(
            [
                {"params": warp_params, "lr": warp_lr},
                {"params": global_affine_params, "lr": affine_lr},
            ]
        ) if warp_params else None

        # warp_start_iter = 6000  # start actually updating warp after 4000 iters
        warp_start_iter = 0  # start actually updating warp after 4000 iters

    # ---------------------------------------------------------------------
    # Load checkpoint (support old 2-tuple and new 3-tuple with warp_state)
    # ---------------------------------------------------------------------
    if checkpoint:
        ckpt = torch.load(checkpoint)
        if isinstance(ckpt, (list, tuple)) and len(ckpt) == 3:
            (model_params, first_iter, warp_state) = ckpt
        else:
            (model_params, first_iter) = ckpt
            warp_state = None
        
        first_iter = 0

        gaussians.restore(model_params, opt)
        if lambda_pos_reg > 0.0:
            with torch.no_grad():
                # adapt this line to your GaussianModel API if needed
                restored_xyz = gaussians.get_xyz.detach().clone()

        if warp_state is not None:
            uid_to_cam = {cam.uid: cam for cam in train_cameras}
            for uid, entry in warp_state.items():
                if uid not in uid_to_cam:
                    continue
                cam = uid_to_cam[uid]
                if isinstance(entry, dict):
                    wf = entry.get('warp_field', None)
                    ga = entry.get('global_affine', None)
                else:
                    # backwards compat: only warp_field
                    wf = entry
                    ga = None

                if wf is not None and cam.warp_field.shape == wf.shape:
                    cam.warp_field.data = wf.to(cam.warp_field.device)
                if ga is not None and cam.global_affine.shape == ga.shape:
                    cam.global_affine.data = ga.to(cam.global_affine.device)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    last_prune_iter = None
    print(prune_sched)

    # Optional Stable Diffusion guidance
    if dataset.lambda_diffusion:
        guidance_sd = StableDiffusion(device="cuda")
        guidance_sd.get_text_embeds([""], [""])
        print(f"[INFO] loaded SD!")

    save_cc = 0
    diff_cam = copy.deepcopy(scene.getTrainCameras()[0])

    for iteration in range(first_iter, opt.iterations + 1):
        save_cc += 1
        iter_start.record()

        if iteration>4000:
            gaussians.update_learning_rate(iteration)
        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random training camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_idxs = list(np.arange(len(viewpoint_stack)))
        rand = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand)
        viewpoint_idx = viewpoint_idxs.pop(rand)

        # Optional diffusion camera
        pick_diff_cam = (
            (randint(1, 100) <= (args.SDS_freq * 100)) and
            dataset.lambda_diffusion and
            iteration > (opt.iterations * 2 / 3)
        )
        if pick_diff_cam:
            diff_pose = scene.getRandEllipsePose(viewpoint_idx, 0, z_variation=0)
            diff_cam.world_view_transform = torch.tensor(
                getWorld2View2(diff_pose[:3, :3].T, diff_pose[:3, 3],
                               diff_cam.trans, diff_cam.scale)
            ).transpose(0, 1).cuda()
            diff_cam.full_proj_transform = (
                diff_cam.world_view_transform.unsqueeze(0).bmm(
                    diff_cam.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            diff_cam.camera_center = diff_cam.world_view_transform.inverse()[3, :3]
            diff_render_pkg = render(diff_cam, gaussians, pipe, background)
            diff_image = diff_render_pkg["render"]

        # Render main training view
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]                                   # [C,H,W]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["depth"]                                    # [1,H,W]

        # -----------------------------------------------------------------
        # Warp GT image and GT depth with global affine + local warp field
        # -----------------------------------------------------------------
        gt_image_raw = viewpoint_cam.original_image.to(image.device)   # [C,H,W]
        gt_depth_raw = viewpoint_cam.depth.to(image.device)            # [H,W] or [1,H,W]

        warped_gt_image, warped_gt_depth, valid_mask = apply_warp(
            gt_image_raw, gt_depth_raw,
            viewpoint_cam.warp_field,
            viewpoint_cam.warp_base_grid,
            viewpoint_cam.global_affine
        )

        # RGB loss (masked L1 + masked SSIM)
        Ll1_img = masked_l1(image, warped_gt_image, valid_mask)
        Lssim_img = masked_ssim(image, warped_gt_image, valid_mask)
        img_loss = (1.0 - opt.lambda_dssim) * Ll1_img + opt.lambda_dssim * (1.0 - Lssim_img)

        loss = img_loss

        diffusion_loss = None
        depth_loss = None
        warp_reg = None
        pos_reg = None    # NEW: position regularization term


        # Depth loss (masked L1 + masked SSIM)
        pred_depth = depth.squeeze(0)                                  # [H,W]
        if warped_gt_depth is not None:
            Ll1_depth = masked_l1(pred_depth, warped_gt_depth, valid_mask)
            Lssim_depth = masked_ssim(pred_depth.unsqueeze(0), warped_gt_depth.unsqueeze(0), valid_mask)
            depth_loss = (1.0 - opt.lambda_dssim) * Ll1_depth + opt.lambda_dssim * (1.0 - Lssim_depth)

            # Use lambda_pearson as global depth weight (reusing config field)
            if getattr(dataset, "lambda_pearson", 0.0) > 0.0:
                loss += dataset.lambda_pearson * depth_loss
            else:
                loss += depth_loss

        # Diffusion guidance
        if pick_diff_cam:
            diffusion_loss = guidance_sd.train_step(diff_image.unsqueeze(0), dataset.step_ratio)
            loss += dataset.lambda_diffusion * diffusion_loss

        # Local warp regularization: high lambdas to keep texture / shape
        # warp_reg = warp_regularization_loss(viewpoint_cam.warp_field,
        #                                     lambda_mag=1e-3,
        #                                     lambda_s1=1e-2,
        #                                     lambda_s2=1e-2)
        warp_reg = warp_regularization_loss(viewpoint_cam.warp_field,
                                            lambda_mag=0.1,
                                            lambda_s1=1,
                                            lambda_s2=1)
        
        loss += warp_reg
        
        warp_reg = warp_regularization_loss(viewpoint_cam.warp_field,
                                            lambda_mag=0.1,
                                            lambda_s1=1,
                                            lambda_s2=1)
        
        loss += warp_reg
        
        # --- NEW: L2 regularization on Gaussian positions ---
        if lambda_pos_reg > 0.0:
            # Current positions
            xyz_now = gaussians.get_xyz

            # If we didn't come from a checkpoint, initialize reference
            # positions at the first iteration.
            if restored_xyz is None:
                with torch.no_grad():
                    restored_xyz = xyz_now.detach().clone()

            # Densification / pruning may change the number of Gaussians;
            # use the overlapping portion to keep it robust.
            N = min(restored_xyz.shape[0], xyz_now.shape[0])
            if N > 0:
                diff = xyz_now[:N] - restored_xyz[:N]
                # Mean L2 over Gaussians
                pos_reg = (diff ** 2).sum(dim=1).mean()
                loss = loss + lambda_pos_reg * pos_reg
                
        loss = loss * 4.0

        
        # -----------------------------------------------
        # Loss guard: skip crazy outlier views
        # -----------------------------------------------
        if loss_guard.should_skip(loss.item()):
            # Optionally log something so you know it's happening
            print(f"[LOSS GUARD] Skipping iter {iteration}: "
                  f"loss={loss.item():.4f}, "
                  f"mean_recent={loss_guard.last_mean:.4f}, "
                  f"ratio={loss_guard.last_ratio:.2f}")
            iter_end.record()
            # Skip this iteration entirely: no backward, no optimizer step,
            # no densification, etc.
            continue

        # Normal case: backprop
        loss.backward()
        iter_end.record()


        with torch.no_grad():
            # Debug comparison every 50 iters: [render | GT | warped GT]
            if iteration % 50 == 0:
                save_debug_comparison(
                    image,
                    gt_image_raw,
                    warped_gt_image,
                    dataset.model_path,
                    iteration,
                    viewpoint_cam.image_name
                )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            losses = [depth_loss, diffusion_loss, warp_reg]
            names = ["Depth", "Diffusion", "Warp Reg"]

            if iteration % 10 == 0:
                postfix_dict = {
                    "EMA Loss": f"{ema_loss_for_log:.7f}",
                    "Total Loss": f"{loss:.7f}",
                }
                for l, n in zip(losses, names):
                    if l is not None:
                        postfix_dict[n] = f"{l:.7f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard & validation
            tr_dict = {names[i]: losses[i] for i in range(len(losses))}
            training_report(
                tb_writer, iteration, Ll1_img, loss, l1_loss,
                tr_dict, iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background)
            )

            # Save model + warped GT images
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                save_warped_gt_images(scene, dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # if iteration < 4000:
                    #     gaussians.densify_and_prune(
                    #         opt.densify_grad_threshold, 0.05,
                    #         scene.cameras_extent, size_threshold
                    #     )
                    # else:
                    #     gaussians.densify_and_prune(
                    #         opt.densify_grad_threshold, 0.3,
                    #         scene.cameras_extent, size_threshold
                    #     )
                    if iteration > 2e4:
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.05,
                            scene.cameras_extent, size_threshold
                        )                        

                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # # Optimizer steps
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none=True)

            #     # Update warp params only after warp_start_iter
            #     if warp_optimizer is not None and iteration >= warp_start_iter:
            #         warp_optimizer.step()
            #         warp_optimizer.zero_grad(set_to_none=True)
            
            # -------------------------------------------------------------
            # After Gaussians unfreeze: reduce warp learning rates by 10×
            # -------------------------------------------------------------
            if (not warp_lr_scaled) and iteration >= freeze_gaussians_until:
                if warp_optimizer is not None:
                    for pg in warp_optimizer.param_groups:
                        pg["lr"] *= 0.03   # make each LR 1/10 of its previous value
                warp_lr_scaled = True
            else:
                if iteration%10000 == 9999:
                    pg["lr"] *= 0.3   # make each LR 1/10 of its previous value



            # Optimizer steps
            if iteration < opt.iterations:
                # --- Freeze Gaussians before freeze_gaussians_until ---
                if iteration >= freeze_gaussians_until:
                    gaussians.optimizer.step()
                # Always clear Gaussian grads so they don't accumulate
                gaussians.optimizer.zero_grad(set_to_none=True)

                # Warp / affine: optimize from warp_start_iter
                if warp_optimizer is not None and iteration >= warp_start_iter:
                    warp_optimizer.step()
                    warp_optimizer.zero_grad(set_to_none=True)


            # Pruning schedule
            if iteration in prune_sched:
                os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
                os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
                scene.save(iteration - 1)
                prune_floaters(
                    scene.getTrainCameras().copy(), gaussians, pipe,
                    background, dataset, iteration
                )
                scene.save(iteration + 1)
                last_prune_iter = iteration

            if last_prune_iter is not None and iteration != last_prune_iter and \
                    iteration - last_prune_iter > dataset.densify_lag and \
                    iteration - last_prune_iter < dataset.densify_period + dataset.densify_lag and \
                    iteration % 100 == 0:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold, 0.01,
                    scene.cameras_extent, 20
                )
                print('Densifying')

            # Checkpoint model + warp params
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                ckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save(
                    (gaussians.capture(), iteration, get_warp_state(scene)),
                    ckpt_path
                )


# -------------------------------------------------------------------------
# Floater pruning (unchanged)
# -------------------------------------------------------------------------

def calc_alpha(means2D, conic_opac, x, y):
    dx = x - means2D[:, 0]
    dy = y - means2D[:, 1]
    power = -0.5 * (conic_opac[:, 0] * (dx * dx) +
                    conic_opac[:, 2] * (dy * dy)) - conic_opac[:, 1] * dx * dy
    alpha = power
    alpha[power > 0] = -100
    return alpha


def prune_floaters(viewpoint_stack, gaussians, pipe, background, dataset, iteration):
    with torch.no_grad():
        os.makedirs(os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"modes_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"depth_{iteration}"), exist_ok=True)
        os.makedirs(os.path.join(dataset.model_path, f"diff_{iteration}"), exist_ok=True)

        mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

        plt.figure(figsize=(25, 20))

        dips = []
        point_lists = []
        means2Ds = []
        conic_opacities = []
        mode_ids = []
        diffs = []
        names = []
        for view in viewpoint_stack:
            names.append(view.image_name)
            render_pkg = render(view, gaussians, pipe, background, ret_pts=True)
            mode_id = render_pkg["mode_id"]
            mode = render_pkg["modes"]
            point_list = render_pkg["point_list"]
            depth = render_pkg["alpha_depth"]
            means2D = render_pkg["means2D"]
            conic_opacity = render_pkg["conic_opacity"]
            diff = calc_diff(mode, depth)

            plt.imsave(
                os.path.join(dataset.model_path, f"modes_{iteration}", f"{view.image_name}.png"),
                mode.cpu().numpy().squeeze(), cmap='jet'
            )
            plt.imsave(
                os.path.join(dataset.model_path, f"depth_{iteration}", f"{view.image_name}.png"),
                depth.cpu().numpy().squeeze(), cmap='jet'
            )

            point_lists.append(point_list)
            means2Ds.append(means2D)
            conic_opacities.append(conic_opacity)
            mode_ids.append(mode_id)
            diffs.append(diff)
            dips.append(diptest.dipstat(diff[diff > 0].cpu().numpy()))

        dips = np.array(dips)
        avg_dip = dips.mean()
        perc = dataset.prune_perc * 100 * np.exp(-1 * dataset.prune_exp * avg_dip)

        if perc < 80:
            perc = 80
        print(f'Percentile {perc}')

        for name, mode_id, point_list, diff, means2D, conic_opacity in zip(
                names, mode_ids, point_lists, diffs, means2Ds, conic_opacities):
            submask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

            diffpos = diff[diff > 0]
            threshold = np.percentile(diffpos.cpu().numpy(), perc)
            pruned_modes_mask = (diff > threshold).squeeze()
            cv2.imwrite(
                os.path.join(dataset.model_path, f"pruned_modes_mask_{iteration}", f"{name}.png"),
                pruned_modes_mask.cpu().numpy().squeeze().astype(np.uint8) * 255
            )

            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(pruned_modes_mask.shape[0]),
                torch.arange(pruned_modes_mask.shape[1]),
                indexing='ij'
            )
            pixel_y = pixel_y.to('cuda')
            pixel_x = pixel_x.to('cuda')
            prune_mode_ids = mode_id[:, pruned_modes_mask]
            pixel_x = pixel_x[pruned_modes_mask]
            pixel_y = pixel_y[pruned_modes_mask]

            neg_mask = (prune_mode_ids == -1).any(dim=0)
            prune_mode_ids = prune_mode_ids[:, ~neg_mask]
            pixel_x = pixel_x[~neg_mask]
            pixel_y = pixel_y[~neg_mask]

            selected_gaussians = set()
            for j in range(prune_mode_ids.shape[-1]):
                x = pixel_x[j]
                y = pixel_y[j]
                gausses = point_list[prune_mode_ids[0, j]:prune_mode_ids[1, j] + 1].long()
                c_opacs = conic_opacity[gausses]
                m2Ds = means2D[gausses]
                test_alpha = calc_alpha(m2Ds, c_opacs, x, y)
                alpha_mask = test_alpha > dataset.power_thresh
                gausses = gausses[alpha_mask]
                selected_gaussians.update(gausses.tolist())

            submask[list(selected_gaussians)] = True
            print(f"submask {torch.count_nonzero(submask)}")

            mask = mask | submask
            num_points_pruned = submask.sum()
            print(f'Pruning {num_points_pruned} gaussians')

        print(gaussians.get_xyz.shape[0])
        gaussians.prune_points(mask)
        print(gaussians.get_xyz.shape[0])


# -------------------------------------------------------------------------
# Logging / reporting
# -------------------------------------------------------------------------

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Tensorboard Found!")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1_img, loss, l1_loss_fn, tr_dict, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        for k, v in tr_dict.items():
            if v is not None:
                tb_writer.add_scalar('train_loss_patches/' + k, v.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_img', Ll1_img.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Validation
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                         for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0, 1.0
                    )
                    gt_image_raw = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )

                    if hasattr(viewpoint, "warp_field") and hasattr(viewpoint, "warp_base_grid") and hasattr(viewpoint, "global_affine"):
                        warped_gt, _, valid_mask = apply_warp(
                            gt_image_raw, None,
                            viewpoint.warp_field,
                            viewpoint.warp_base_grid,
                            viewpoint.global_affine
                        )
                        gt_image = torch.clamp(warped_gt, 0.0, 1.0)
                    else:
                        gt_image = gt_image_raw
                        valid_mask = None

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration
                            )

                    # Masked L1 and PSNR
                    l1_val = masked_l1(image, gt_image, valid_mask)
                    l1_test += l1_val.double()
                    # PSNR still computed globally (optionally could be masked)
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test
                ))
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - l1_loss',
                        l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr',
                        psnr_test, iteration
                    )

        torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 37000, 50_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 37000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--prune_sched", nargs="+", type=int, default=[])
    # parser.add_argument("--SDS_freq", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    dataset = lp.extract(args)

    print("Optimizing " + dataset.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.step,
        args.max_cameras,
        args.prune_sched
    )

    print("\nTraining complete.")
