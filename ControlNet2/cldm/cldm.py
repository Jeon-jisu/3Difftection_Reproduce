import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class EpipolarWarpOperator(nn.Module):
    def __init__(self):
        super(EpipolarWarpOperator, self).__init__()
        self.conv1 = None
        # self.conv1 = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, source_intrinsics, target_intrinsics, source_pose, target_pose):
        dtype = x.dtype
        source_intrinsics = source_intrinsics.to(dtype)
        target_intrinsics = target_intrinsics.to(dtype)
        source_pose = source_pose.to(dtype)
        target_pose = target_pose.to(dtype)
        batch_size, channels, height, width = x.size()
        if self.conv1 is None or self.conv1.in_channels != channels:
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1).to(x.device)
        
        # Compute fundamental matrix for each sample in the batch
        F_batch = self.compute_fundamental_matrix_batch(source_intrinsics, target_intrinsics, source_pose, target_pose)
        
        # Generate pixel coordinates
        pixel_coords = self.generate_pixel_coords(batch_size, height, width, x.device)
        
        # Compute epipolar lines for each pixel in each sample
        epipolar_lines = torch.bmm(F_batch.transpose(1, 2), pixel_coords)
        
        # Sample features along epipolar lines
        sampled_features = self.sample_along_epipolar_lines(x, epipolar_lines, height, width)
        
        # Apply convolution and activation
        output = self.conv1(sampled_features)
        output = self.relu(output)
        
        return output

    def compute_fundamental_matrix_batch(self, source_intrinsics, target_intrinsics, source_pose, target_pose):
        # print("source_intrinsics",source_intrinsics)
        batch_size = source_intrinsics.shape[0]
        F_batch = []
        for i in range(batch_size):
            K_source = source_intrinsics[i]
            K_target = target_intrinsics[i]
            R_source = self.rotation_vector_to_matrix(source_pose[i, :3])
            t_source = source_pose[i, 3:].unsqueeze(1)
            R_target = self.rotation_vector_to_matrix(target_pose[i, :3])
            t_target = target_pose[i, 3:].unsqueeze(1)
            
            R_relative = torch.mm(R_source.t(), R_target)
            t_relative = t_source - torch.mm(R_relative, t_target)
            
            E = torch.mm(self.skew_symmetric(t_relative.squeeze()), R_relative)
            F = torch.mm(torch.mm(torch.inverse(K_target).t(), E), torch.inverse(K_source))
            F_batch.append(F)
        
        return torch.stack(F_batch)

    def generate_pixel_coords(self, batch_size, height, width, device):
        x = torch.arange(width, device=device).float()
        y = torch.arange(height, device=device).float()
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x)
        pixel_coords = torch.stack((grid_x.flatten(), grid_y.flatten(), ones.flatten()), dim=0)
        pixel_coords = pixel_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        return pixel_coords

    def sample_along_epipolar_lines(self, x, epipolar_lines, height, width):
        batch_size, channels, _, _ = x.shape
        sampled_features = []
        y = torch.arange(height, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
        y = y.expand(batch_size, 1, -1, 1)  # [batch_size, 1, height, 1]
        for i in range(width):
            for j in range(height):
                l = epipolar_lines[:, :, i*height + j].view(batch_size, 3, 1)
                a, b, c = l[:, 0], l[:, 1], l[:, 2]

                # 에피폴라 선과 이미지 경계의 교차점 계산
                x1 = torch.clamp(-c / (a + 1e-10), 0, width - 1)
                x2 = torch.clamp(-(b*(height-1) + c) / (a + 1e-10), 0, width - 1)
                y1 = torch.clamp(-c / (b + 1e-10), 0, height - 1)
                y2 = torch.clamp(-(a*(width-1) + c) / (b + 1e-10), 0, height - 1)

                # 에피폴라 선을 따라 일정 간격으로 샘플링 포인트 생성
                num_samples = 20  # 샘플링 포인트 수
                t = torch.linspace(0, 1, num_samples, device=x.device).view(1, -1).expand(batch_size, -1)
                sample_x = x1.view(-1, 1) * (1 - t) + x2.view(-1, 1) * t
                sample_y = y1.view(-1, 1) * (1 - t) + y2.view(-1, 1) * t

                # 정규화된 좌표로 변환
                grid = torch.stack((
                    2 * sample_x / (width - 1) - 1,
                    2 * sample_y / (height - 1) - 1
                ), dim=-1).view(batch_size, 1, -1, 2)

                # 샘플링 및 특징 추출
                sampled = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
                
                # 샘플링된 특징들의 평균 계산
                averaged_feature = sampled.mean(dim=3)  # [batch_size, channels, 1]
                sampled_features.append(averaged_feature.squeeze(2))

        # print("batch_size",batch_size,"channels",channels,"height",height,"width",width)
        return torch.stack(sampled_features, dim=2).view(batch_size, channels, height, width)

    @staticmethod
    def skew_symmetric(v):
        return torch.tensor([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]], device=v.device)

    @staticmethod
    def rotation_vector_to_matrix(rotation_vector):
        theta = torch.norm(rotation_vector)
        if theta < 1e-6:
            return torch.eye(3, device=rotation_vector.device)
        
        r = rotation_vector / theta
        I = torch.eye(3, device=rotation_vector.device)
        r_cross = torch.tensor([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ], device=rotation_vector.device)
        
        rotation_matrix = torch.cos(theta) * I + (1 - torch.cos(theta)) * torch.outer(r, r) + torch.sin(theta) * r_cross
        return rotation_matrix

    
class ControlledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, source_pose=None, target_pose=None, source_intrinsic=None, target_intrinsic=None, **kwargs):
        hs = []
        # 이 부분은 Frozen된 SD의 부분인듯. 
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                # hs에는 target image가 time과 context를 고려한 다양한 해상도의 값을 가지고 있겠다. 
            h = self.middle_block(h, emb, context) # 마지막에는 Middel Block을 거쳐서 나온 값.

        if control is not None:
            h += control.pop()
        # output blocks를 순회하면서 디코딩을 수행. hs에는 앞서 인코딩할때 각 해상도에서 저장해놓은 feature map이 있음. 
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # print(f"h shape: {h.shape}")
                hs_popped = hs.pop()
                # print(f"hs.pop() shape: {hs_popped.shape}")
                control_popped = control.pop()
                # print(f"control.pop() shape: {control_popped.shape}")
                h = torch.cat([h, hs_popped + control_popped], dim=1)
            h = module(h, emb, context)
            # print(f"After concatenation, h shape: {h.shape}")
            # Apply epipolar warping only to the last two decoder blocks
            
            # print(f"After epipolar_warp, h shape: {h.shape}")
        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.epipolar_warp = EpipolarWarpOperator()

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, source_pose=None, target_pose=None, source_intrinsic=None, target_intrinsic=None,**kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None # 첫번째 encoder block에서만 통합하여 들어간다. 
            else:
                h = module(h, emb, context)
            # print("epipolar로 잘 전달되는지 확인",target_pose[-1])
            h = self.epipolar_warp(h, source_intrinsic, target_intrinsic, source_pose, target_pose)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
            
        # print("control:",control)
        # source_pose, target_pose, intrinsic_params를 batch에서 가져옵니다.
        source_pose = batch.get('source_camera_pose').to(self.device)
        target_pose = batch.get('target_camera_pose').to(self.device)
        source_intrinsic = batch.get('source_camera_intrinsic').to(self.device)
        target_intrinsic = batch.get('target_camera_intrinsic').to(self.device)
        # print("Debug - get_input - source_pose:", source_pose)
        # print("Debug - get_input - target_pose:", target_pose)
        # print("Debug - get_input - source_intrinsic:", source_intrinsic)
        # print("Debug - get_input - target_intrinsic:", target_intrinsic)
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control],source_pose=source_pose, target_pose=target_pose, 
                       source_intrinsic=source_intrinsic, target_intrinsic = target_intrinsic)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # 여기에서 source_pose, target_pose, intrinsic_params를 가져옵니다.
            source_pose = cond.get('source_pose')
            target_pose = cond.get('target_pose')
            source_intrinsic = cond.get('source_intrinsic')
            target_intrinsic = cond.get('target_intrinsic')
            # print("Debug - source_pose:", source_pose)
            # print("Debug - target_pose:", target_pose)
            # print("Debug - source_intrinsic:", source_intrinsic)
            # print("Debug - target_intrinsic:", target_intrinsic)
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,source_pose=source_pose, target_pose=target_pose, 
                                  source_intrinsic=source_intrinsic, target_intrinsic = target_intrinsic)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control,source_pose=source_pose, target_pose=target_pose, 
                                  source_intrinsic=source_intrinsic, target_intrinsic = target_intrinsic)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c_dict = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c_dict["c_concat"][0][:N], c_dict["c_crossattn"][0][:N]
        # print("****",c_dict["source_pose"])
        source_pose = c_dict["source_pose"]
        target_pose = c_dict["target_pose"]
        source_intrinsic = c_dict["source_intrinsic"]
        target_intrinsic = c_dict["target_intrinsic"]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],
                                                           "source_pose": source_pose,
                                                        "target_pose": target_pose,
                                                        "source_intrinsic": source_intrinsic,
                                                        "target_intrinsic": target_intrinsic},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 0.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross],"source_pose": source_pose,
                                                        "target_pose": target_pose,
                                                        "source_intrinsic": source_intrinsic,
                                                        "target_intrinsic": target_intrinsic}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],"source_pose": source_pose,
                                                        "target_pose": target_pose,
                                                        "source_intrinsic": source_intrinsic,
                                                        "target_intrinsic": target_intrinsic},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        
        # print("Debug - sample_log - cond keys:", cond.keys())
        # print("Debug - sample_log - source_pose:", cond.get('source_pose'))
        # print("Debug - sample_log - target_pose:", cond.get('target_pose'))
        # print("Debug - sample_log - source_intrinsic:", cond.get('source_intrinsic'))
        # print("Debug - sample_log - target_intrinsic:", cond.get('target_intrinsic'))
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()