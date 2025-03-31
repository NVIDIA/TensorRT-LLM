import numpy as np
import torch
from tqdm.auto import tqdm


def space_timesteps(num_timesteps, timestep_respacing):
    if num_timesteps < timestep_respacing:
        raise ValueError(
            f"cannot divide section of {num_timesteps} steps into {timestep_respacing}"
        )
    frac_stride = (num_timesteps - 1) / (timestep_respacing - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(timestep_respacing):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return set(taken_steps)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class DiTDiffusionPipeline:

    def __init__(self, dit_model, timestep_respacing, diffusion_steps=1000):
        self.dit_model = dit_model
        self.timesteps = space_timesteps(diffusion_steps, timestep_respacing)
        self.timestep_map = []
        self.original_num_steps = diffusion_steps
        betas = self._setup_betas()
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1],
                      self.posterior_variance[1:])) if len(
                          self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))

    def _setup_betas(self):
        scale = 1000 / self.original_num_steps
        betas = np.linspace(scale * 0.0001,
                            scale * 0.02,
                            self.original_num_steps,
                            dtype=np.float64)
        last_alpha_cumprod = 1.0
        new_betas = []
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in self.timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        betas = np.array(new_betas)
        return betas

    def run(self, sample, labels):
        indices = list(range(self.num_timesteps))[::-1]

        for i in tqdm(indices):
            t = torch.tensor([i] * sample.shape[0], device=sample.device)
            B, C = sample.shape[:2]
            assert t.shape == (B, )
            map_tensor = torch.tensor(self.timestep_map,
                                      device=t.device,
                                      dtype=t.dtype)
            model_output = self.dit_model(sample, map_tensor[t], labels)
            assert model_output.shape == (B, C * 2, *sample.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped,
                                           t, sample.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, sample.shape)

            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            pred_xstart = (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     sample.shape) * sample -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     sample.shape) * model_output)

            model_mean = (_extract_into_tensor(self.posterior_mean_coef1, t,
                                               sample.shape) * pred_xstart +
                          _extract_into_tensor(self.posterior_mean_coef2, t,
                                               sample.shape) * sample)

            assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == sample.shape
            noise = torch.randn_like(sample)
            nonzero_mask = ((t != 0).float().view(
                -1, *([1] * (len(sample.shape) - 1))))
            sample = model_mean + nonzero_mask * torch.exp(
                0.5 * model_log_variance) * noise
        return sample
