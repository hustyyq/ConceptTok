# Concept Tokenizer note: adapted from REPA (https://github.com/sihyun-yu/REPA) for REPA-SAE concept guidance experiments.
import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            topk=64,
            use_sae=False,
            distill_target="feature",
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.topk = topk
        self.use_sae = use_sae
        self.distill_target = distill_target
        self._printed_token_mismatch = False
        self._printed_head_mismatch = False

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        if self.distill_target == "attn":
            model_output, zs_tilde = model(
                model_input,
                time_input.flatten(),
                return_encoder_attn=True,
                **model_kwargs,
            )
        else:
            model_output, zs_tilde = model(
                model_input,
                time_input.flatten(),
                **model_kwargs,
            )
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss
        if self.distill_target == "attn":
            if zs is None or len(zs) == 0:
                raise ValueError("Attention distillation requires teacher attention tensors in zs.")
            teacher_attn = zs[0]
            student_attn = zs_tilde

            # Align token dimensions without assuming a specific special-token layout.
            # For ViTs, special tokens are typically prepended, so we keep the tail (patch tokens).
            if teacher_attn.shape[-1] != student_attn.shape[-1]:
                if not self._printed_token_mismatch:
                    print(
                        f"[attn-distill] token mismatch: teacher={teacher_attn.shape[-1]} "
                        f"student={student_attn.shape[-1]}, aligning by tail-cropping."
                    )
                    self._printed_token_mismatch = True
                token_count = min(teacher_attn.shape[-1], student_attn.shape[-1])
                teacher_attn = teacher_attn[:, :, -token_count:, -token_count:]
                student_attn = student_attn[:, :, -token_count:, -token_count:]

            # If head count mismatches, compare the average attention pattern.
            if teacher_attn.shape[1] != student_attn.shape[1]:
                if not self._printed_head_mismatch:
                    print(
                        f"[attn-distill] head mismatch: teacher={teacher_attn.shape[1]} "
                        f"student={student_attn.shape[1]}, aligning by head-mean."
                    )
                    self._printed_head_mismatch = True
                teacher_attn = teacher_attn.mean(dim=1, keepdim=True)
                student_attn = student_attn.mean(dim=1, keepdim=True)

            eps = 1e-8
            teacher_attn = teacher_attn.detach()
            # Cropping may remove probability mass; renormalize to valid distributions.
            teacher_attn = teacher_attn / teacher_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
            student_attn = student_attn / student_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
            proj_loss = mean_flat(
                F.kl_div(
                    student_attn.clamp_min(eps).log(),
                    teacher_attn.clamp_min(eps),
                    reduction="none",
                )
            )

        elif not self.use_sae:
            proj_loss = 0.
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
            proj_loss /= (len(zs) * bsz)

        else:
            proj_loss = 0.0
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                # Support both [B, D] and [B, T, D]: flatten to [N, D]
                if z.dim() == 3:
                    B, T, D = z.shape
                    z_flat = z.view(B * T, D)
                    z_tilde_flat = z_tilde.view(B * T, D)
                else:
                    z_flat = z
                    z_tilde_flat = z_tilde
                # top-k smoothed target from teacher z
                with torch.no_grad():
                    topk_idx = torch.topk(z_flat, self.topk, dim=1).indices  # [N, topk]
                    smoothed_labels = torch.zeros_like(z_flat)
                    smoothed_labels.scatter_(1, topk_idx, 1.0 / self.topk)
                log_prob = F.log_softmax(z_tilde_flat, dim=1)
                loss = -torch.sum(log_prob * smoothed_labels, dim=1).mean()
                proj_loss += loss
            proj_loss = proj_loss / len(zs)

        return denoising_loss, proj_loss
