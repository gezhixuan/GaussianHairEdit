import copy
from dataclasses import dataclass, field

from tqdm import tqdm

import torch
import threestudio
import os

# from threestudio.utils.clip_metrics import ClipSimilarity
from threestudio.systems.GaussianEditor import GaussianEditor
import cv2
import numpy as np

# debug_root = "./debug_Someone"
# debug_root = "./debug_ZhixuanV3.0"
# debug_root = "./debug_Zhixuan"
debug_root = "./debug_Liza_new"


@threestudio.register("gsedit-system-edit")
class VcEdit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

        # Face consistency related configs (we now use "skin" as the main prompt)
        preserve_face: bool = False
        face_seg_prompt: str = "skin"   # used as skin prompt
        hair_seg_prompt: str = "hair"   # kept for compatibility, not used

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join(
                "edit_cache", self.cfg.gs_source.replace("/", "-")
            )

        self.bg_gaussian = None

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.origin_frames = self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            fg_mask = self.update_mask()
            self.bg_gaussian = copy.deepcopy(self.gaussian)
            self.bg_gaussian.prune_points(fg_mask)

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )

    # ===== Build skin mask using a single "skin" prompt =====
    @torch.no_grad()
    def _build_face_mask(
        self,
        img: torch.Tensor,
        skin_prompt: str,
        debug_prefix: str = "",
    ) -> torch.Tensor:
        """
        img: (H, W, 3) or (1, H, W, 3), on CUDA.
        Returns: bool tensor of shape (H, W), representing a "skin" region mask.

        If debug_prefix is not empty, also saves a 1x2 panel:
        [input image | skin mask]
        under: {debug_root}/mask_debug/{debug_prefix}.png
        """
        # Ensure batch dimension
        if img.dim() == 3:
            img_b = img[None]  # (1, H, W, 3)
        else:
            img_b = img        # (1, H, W, 3)

        # Use float32 for segmentation to avoid AMP half issues
        img_b = img_b.float()

        # Run segmentor with "skin" prompt
        skin_score = self.text_segmentor(img_b, skin_prompt)[0]  # (1, H, W)
        skin_mask = skin_score[0].bool()                         # (H, W)

        # Optional debug visualization
        if debug_prefix != "":
            # Input image: (1, H, W, 3) -> (H, W, 3)
            img_vis = img_b[0].detach().cpu().numpy()
            img_vis = (img_vis.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

            # Mask: (H, W) bool -> (H, W, 3) uint8
            mask_np = (
                skin_mask.float().unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()
            )
            mask_np = (mask_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)

            panel = np.concatenate([img_vis, mask_np], axis=1)

            debug_dir = os.path.join(debug_root, "mask_debug")
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"{debug_prefix}.png")
            cv2.imwrite(out_path, panel)

        return skin_mask

    # ===== Apply face-preserving operation on guidance outputs =====
    @torch.no_grad()
    def _preserve_face_identity_in_result(self, curr_frames, result):
        """
        curr_frames: output of render_all_view_no_cache,
                     dict[view_idx] -> (1, H, W, 3)
        result: guidance output dict, containing result["edit_images"][idx]
        Behavior:
            For each view, on the intersection of "skin" masks
            between pre-guidance and post-guidance, copy pixels from
            pre-guidance image to post-guidance image.
            Also writes per-view mask debug panels for pre/post images.
        """
        if not getattr(self.cfg, "preserve_face", False):
            return

        for idx, view_idx in enumerate(self.view_list):
            if view_idx not in curr_frames:
                continue

            # Normalize view id for naming
            if isinstance(view_idx, int):
                view_id = view_idx
            else:
                view_id = int(view_idx)

            # Pre-guidance image, usually float32, (H, W, 3)
            before = curr_frames[view_idx][0]  # CUDA

            # Post-guidance image (may be HWC or CHW, often float16)
            after = result["edit_images"][idx]  # CUDA

            chw = False
            if after.dim() == 3 and after.shape[0] == 3 and after.shape[-1] != 3:
                # Assume CHW
                after_hwc = after.permute(1, 2, 0)  # -> (H, W, 3)
                chw = True
            else:
                after_hwc = after  # Already (H, W, 3)

            # Build skin masks for pre and post images, with debug panels
            mask_before = self._build_face_mask(
                before,
                self.cfg.face_seg_prompt,
                debug_prefix=f"step{self.global_step:06d}_view{view_id:04d}_pre",
            )
            mask_after = self._build_face_mask(
                after_hwc,
                self.cfg.face_seg_prompt,
                debug_prefix=f"step{self.global_step:06d}_view{view_id:04d}_post",
            )

            # Intersection of skin regions
            mask_inter = mask_before & mask_after  # (H, W)
            if mask_inter.sum() == 0:
                # If there is no intersection, skip this view
                continue

            mask3 = mask_inter[..., None].expand_as(before)  # (H, W, 3)

            # Ensure source and destination dtypes match
            mixed = after_hwc.clone()                   # destination, e.g. half
            before_cast = before.to(mixed.dtype)        # source cast to same dtype

            mixed[mask3] = before_cast[mask3]           # copy pre-guidance pixels

            if chw:
                mixed = mixed.permute(2, 0, 1)  # back to (3, H, W) if needed

            result["edit_images"][idx] = mixed

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]

        out = self(batch, local=self.cfg.local_edit)
        images = out["comp_rgb"]

        loss = 0.0
        # NeRF-to-NeRF style image reconstruction loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:

            if self.global_step % self.cfg.per_editing_step == 0:
                print("Update guidance at global step: ", self.global_step)
                curr_frames = self.render_all_view_no_cache(
                    gaussian=copy.deepcopy(self.gaussian), with_semantic=False
                )

                result = self.guidance(
                    curr_frames,
                    copy.deepcopy(self.gaussian),
                    self.trainer.datamodule.train_dataset.scene.cameras,
                    self.pipe,
                    self.view_list,
                    calling_idx=self.global_step // self.cfg.per_editing_step,
                )

                # Keep a copy of raw guidance outputs (before face preservation)
                raw_edit_images = [
                    img.detach().clone() for img in result["edit_images"]
                ]

                # Optionally enforce face consistency on result["edit_images"]
                if getattr(self.cfg, "preserve_face", False):
                    self._preserve_face_identity_in_result(curr_frames, result)

                # Use (possibly face-preserved) edit_images as GT for training
                for idx, view_idx in enumerate(self.view_list):
                    self.edit_frames[view_idx] = result["edit_images"][idx][None]

                # ----- DEBUG: save 2x3 grid for each view -----
                # Row 1: pre-guidance image | raw guidance output | face-preserved output
                # Row 2: pre skin mask     | post skin mask      | intersection mask
                step_dir = os.path.join(debug_root, str(self.global_step))
                os.makedirs(step_dir, exist_ok=True)

                for idx, view_idx in enumerate(self.view_list):
                    # Normalize view id
                    if isinstance(view_idx, int):
                        view_id = view_idx
                    else:
                        view_id = int(view_idx)

                    # 1) Pre-guidance image (CUDA, HWC)
                    before_t = curr_frames[view_idx][0].detach()

                    # 2) Raw guidance output before face preservation
                    after_raw_t = raw_edit_images[idx].detach()
                    if (
                        after_raw_t.dim() == 3
                        and after_raw_t.shape[0] == 3
                        and after_raw_t.shape[-1] != 3
                    ):
                        after_raw_hwc_t = after_raw_t.permute(1, 2, 0)
                    else:
                        after_raw_hwc_t = after_raw_t

                    # 3) Face-preserved guidance output (current result["edit_images"])
                    after_new_t = result["edit_images"][idx].detach()
                    if (
                        after_new_t.dim() == 3
                        and after_new_t.shape[0] == 3
                        and after_new_t.shape[-1] != 3
                    ):
                        after_new_hwc_t = after_new_t.permute(1, 2, 0)
                    else:
                        after_new_hwc_t = after_new_t

                    # Build skin masks for pre and raw post images (no extra debug here)
                    mask_before = self._build_face_mask(
                        before_t,
                        self.cfg.face_seg_prompt,
                    )
                    mask_after = self._build_face_mask(
                        after_raw_hwc_t,
                        self.cfg.face_seg_prompt,
                    )
                    mask_inter = mask_before & mask_after

                    # Convert images to numpy (H, W, 3)
                    before = before_t.cpu().numpy()
                    after_raw = after_raw_hwc_t.cpu().numpy()
                    after_new = after_new_hwc_t.cpu().numpy()

                    # Convert masks to RGB-like numpy for visualization
                    def mask_to_rgb(m: torch.Tensor) -> np.ndarray:
                        m_np = m.float().unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()
                        return (m_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)

                    mask_before_img = mask_to_rgb(mask_before)
                    mask_after_img = mask_to_rgb(mask_after)
                    mask_inter_img = mask_to_rgb(mask_inter)

                    # First row: pre | post(raw) | new
                    row1 = np.concatenate([before, after_raw, after_new], axis=1)
                    # Second row: pre mask | post mask | intersection mask
                    row2 = np.concatenate(
                        [mask_before_img, mask_after_img, mask_inter_img], axis=1
                    )

                    comp = np.concatenate([row1, row2], axis=0)
                    comp = (comp.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                    comp = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        os.path.join(step_dir, f"guidance_view_{view_id:04d}.png"),
                        comp,
                    )
                # ----- DEBUG end -----

                concat_edit_frames = (
                    result["edit_images"]
                    .permute(1, 0, 2, 3)
                    .flatten(1, 2)
                    .cpu()
                    .numpy()
                )
                concat_edit_frames = (
                    concat_edit_frames.clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                concat_edit_frames = cv2.cvtColor(
                    concat_edit_frames, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(
                    self.get_save_path(
                        f"edit_images_{self.guidance.tgt_prompt}_"
                        f"{self.cfg.per_editing_step}_{self.global_step}.png"
                    ),
                    concat_edit_frames,
                )

            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            # ----- DEBUG: on non-guidance steps, save render vs GT comparison -----
            if self.global_step % self.cfg.per_editing_step != 0:
                images_np = images.detach().cpu().numpy()
                gt_np = gt_images.detach().cpu().numpy()

                for b, cur_index in enumerate(batch_index):
                    if isinstance(cur_index, int):
                        view_id = cur_index
                    else:
                        view_id = int(cur_index)

                    pred = images_np[b]   # H W 3, render
                    target = gt_np[b]     # H W 3, GT from edit_frames

                    comp = np.concatenate([pred, target], axis=1)
                    comp = (comp.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                    comp = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

                    filename = f"train_step_{self.global_step:06d}_view_{view_id:04d}.png"
                    cv2.imwrite(os.path.join(debug_root, filename), comp)
            # ----- DEBUG end -----

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # DDS loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # Anchor loss on Gaussian parameters
        if (
            self.cfg.loss.lambda_anchor_color > 0
            or self.cfg.loss.lambda_anchor_geo > 0
            or self.cfg.loss.lambda_anchor_scale > 0
            or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # Log all loss hyperparameters for inspection
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(
                    self.origin_frames[id].permute(0, 3, 1, 2),
                    out.permute(0, 3, 1, 2),
                    self.cfg.clip_prompt_origin,
                    self.cfg.clip_prompt_target,
                )
                total_cos += abs(cos_sim.item())
        print(
            self.cfg.clip_prompt_origin,
            self.cfg.clip_prompt_target,
            total_cos / len(self.view_list),
        )
        self.log("train/clip_sim", total_cos / len(self.view_list))

    def render_all_view_with_aug(self):
        self.bg_aug_renders = {}
        for id in tqdm(self.view_list):
            cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
            cur_batch = {
                "index": id,
                "camera": [cur_cam],
                "height": self.trainer.datamodule.train_dataset.height,
                "width": self.trainer.datamodule.train_dataset.width,
            }

            view_out = []
            for bg_color in self.bg_aug_colors:
                out = self(cur_batch, renderbackground=bg_color)["comp_rgb"]
                view_out.append(out[0][None])
            self.bg_aug_renders[id] = torch.cat(view_out, dim=0)

    def multi_view_sync_attn(self, view_list, name):
        processors = self.guidance.pipe.unet.attn_processors
        for _, processor in processors.items():
            all_states = [
                torch.stack(processor.state[name][i], dim=0)
                for i in range(len(processor.state[name]))
            ]
            all_states = torch.stack(all_states, dim=0).permute(1, 0, 2, 3, 4)
            assert len(all_states[0]) == len(view_list)
            all_states = all_states.view(-1, -1, -1, 64, 64, -1)
