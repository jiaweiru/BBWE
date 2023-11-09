import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms


EPSILON = torch.finfo(torch.float32).eps


def replace_denormals(x, eps):
    """Replace numbers close to zero to avoid NaNs in `angle` or `atan2`"""
    y = x.clone()
    y[torch.abs(x) < eps] = eps
    return y


# Calculating losses in the time domain
# Recalculating losses from the time domain preserves STFT consistency
class MagLoss(nn.Module):
    def __init__(self, fft_len, win_inc, win_len, win_fn, loss_tp="l2"):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = win_fn(self.win_len)
        if loss_tp == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        elif loss_tp == "l1":
            self.loss_func = nn.L1Loss(reduction="none")

    def forward(self, ref_wav, est_wav, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        ref_spec = torch.stft(
            ref_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        est_spec = torch.stft(
            est_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        # Log-compressed magnitude
        ref_mag_comp = torch.log1p(
            torch.sqrt(torch.clamp(ref_spec.real**2 + ref_spec.imag**2, EPSILON))
        )
        est_mag_comp = torch.log1p(
            torch.sqrt(torch.clamp(est_spec.real**2 + est_spec.imag**2, EPSILON))
        )

        loss_mag = self.loss_func(ref_mag_comp, est_mag_comp)
        if reduction == "mean":
            loss_mag = torch.mean(loss_mag)
        elif reduction == "batch":
            loss_mag = torch.mean(loss_mag, dim=[1, 2])  # maintain the batch dim

        return loss_mag

    def forward_res(
        self, lr_wav, ref_wav, est_wav, lr=8000, hr=16000, reduction="mean"
    ):
        up_wav = torchaudio.functional.resample(lr_wav, lr, hr)
        ref_wav = ref_wav - up_wav
        est_wav = est_wav - up_wav

        return self(ref_wav, est_wav, reduction)


class RILoss(nn.Module):
    def __init__(self, fft_len, win_inc, win_len, win_fn, alpha=0.5, loss_tp="l2"):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = win_fn(self.win_len)
        self.alpha = alpha
        if loss_tp == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        elif loss_tp == "l1":
            self.loss_func = nn.L1Loss(reduction="none")

    def forward(self, ref_wav, est_wav, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        ref_spec = torch.stft(
            ref_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        est_spec = torch.stft(
            est_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        est_mag = torch.sqrt(
            torch.clamp(est_spec.real**2 + est_spec.imag**2, EPSILON)
        )
        est_phase = torch.arctan2(
            replace_denormals(est_spec.imag, EPSILON),
            replace_denormals(est_spec.real, EPSILON),
        )
        est_mag_comp = torch.pow(est_mag, self.alpha)
        est_spec_comp = torch.stack(
            [
                est_mag_comp * torch.cos(est_phase),
                est_mag_comp * torch.sin(est_phase),
            ],
            dim=1,
        )

        ref_mag = torch.sqrt(
            torch.clamp(ref_spec.real**2 + ref_spec.imag**2, EPSILON)
        )
        ref_phase = torch.arctan2(
            replace_denormals(ref_spec.imag, EPSILON),
            replace_denormals(ref_spec.real, EPSILON),
        )
        ref_mag_comp = torch.pow(ref_mag, self.alpha)
        ref_spec_comp = torch.stack(
            [
                ref_mag_comp * torch.cos(ref_phase),
                ref_mag_comp * torch.sin(ref_phase),
            ],
            dim=1,
        )

        loss = self.loss_func(ref_spec_comp, est_spec_comp)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "batch":
            loss = torch.mean(loss, dim=[1, 2, 3])  # maintain the batch dim

        return loss


class MagRILoss(nn.Module):
    def __init__(self, fft_len, win_inc, win_len, win_fn, alpha=0.5, loss_tp="l2"):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = win_fn(self.win_len)
        self.alpha = alpha
        if loss_tp == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        elif loss_tp == "l1":
            self.loss_func = nn.L1Loss(reduction="none")

    def forward(self, ref_wav, est_wav, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        ref_spec = torch.stft(
            ref_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        est_spec = torch.stft(
            est_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        est_mag = torch.sqrt(
            torch.clamp(est_spec.real**2 + est_spec.imag**2, EPSILON)
        )
        est_phase = torch.arctan2(
            replace_denormals(est_spec.imag, EPSILON),
            replace_denormals(est_spec.real, EPSILON),
        )
        est_mag_comp = torch.pow(est_mag, self.alpha)
        est_spec_comp = torch.stack(
            [
                est_mag_comp * torch.cos(est_phase),
                est_mag_comp * torch.sin(est_phase),
            ],
            dim=1,
        )

        ref_mag = torch.sqrt(
            torch.clamp(ref_spec.real**2 + ref_spec.imag**2, EPSILON)
        )
        ref_phase = torch.arctan2(
            replace_denormals(ref_spec.imag, EPSILON),
            replace_denormals(ref_spec.real, EPSILON),
        )
        ref_mag_comp = torch.pow(ref_mag, self.alpha)
        ref_spec_comp = torch.stack(
            [
                ref_mag_comp * torch.cos(ref_phase),
                ref_mag_comp * torch.sin(ref_phase),
            ],
            dim=1,
        )

        loss_r = self.loss_func(ref_spec_comp[:, 0], est_spec_comp[:, 0])
        loss_i = self.loss_func(ref_spec_comp[:, 1], est_spec_comp[:, 1])
        loss_mag = self.loss_func(ref_mag_comp, est_mag_comp)
        if reduction == "mean":
            loss = torch.mean(loss_r) + torch.mean(loss_i) + torch.mean(loss_mag)
        elif reduction == "batch":
            loss = (
                torch.mean(loss_r, dim=[1, 2])
                + torch.mean(loss_i, dim=[1, 2])
                + torch.mean(loss_mag, dim=[1, 2])
            )  # maintain the batch dim

        return loss


def snr_loss(y_pred_batch, y_true_batch, lens, reduction="mean"):
    assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]
    SNR = torch.zeros(batch_size)

    for i in range(0, batch_size):  # Run over mini-batches
        s_target = y_true_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]
        s_estimate = y_pred_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]

        snr_beforelog = torch.sum(s_target**2, dim=0) / (
            torch.sum((s_estimate - s_target) ** 2, dim=0) + EPSILON
        )
        SNR[i] = 10 * torch.log10(snr_beforelog + EPSILON)

    if reduction == "mean":
        return -SNR.mean()

    return -SNR


def dynamic_range_compression(x, C=1):
    """Dynamique range compression for audio signals"""
    return torch.log(torch.clamp(x, min=EPSILON) * C)


def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal"""
    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf
    Note : L1 loss helps leaning details compared with L2 loss"""

    def __init__(
        self,
        sample_rate=22050,
        hop_length=256,
        win_length=24,
        n_mel_channels=80,
        n_fft=1024,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        mel_normalized=False,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
        dynamic_range_compression=True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_normalized = mel_normalized
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale
        self.dynamic_range_compression = dynamic_range_compression
        self.loss_func = nn.L1Loss(reduction="none")

    def forward(self, y_hat, y, reduction="mean"):
        """Returns L1 Loss over Melspectrograms"""
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        y_hat_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y_hat,
        )
        # y_M = mel_spectogram(self.mel_params, y)
        y_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y,
        )

        # magnitude loss
        loss_mag = self.loss_func(y_M, y_hat_M)

        if reduction == "mean":
            return torch.mean(loss_mag)
        elif reduction == "batch":
            return torch.mean(loss_mag, dim=[1, 2])


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, fft_len, win_inc, win_len, win_fn=torch.hann_window):
        super().__init__()
        self.fft_len = fft_len
        self.win_inc = win_inc
        self.win_len = win_len
        self.window = win_fn(self.win_len)
        self.loss_m = nn.L1Loss(reduction="none")

    def forward(self, ref_wav, est_wav, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        ref_spec = torch.stft(
            ref_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        est_spec = torch.stft(
            est_wav,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(ref_wav.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        ref_mag = torch.sqrt(
            torch.clamp(ref_spec.real**2 + ref_spec.imag**2, EPSILON)
        )
        est_mag = torch.sqrt(
            torch.clamp(est_spec.real**2 + est_spec.imag**2, EPSILON)
        )
        if reduction == "mean":
            # magnitude loss
            loss_mag = torch.mean(self.loss_m(torch.log(ref_mag), torch.log(est_mag)))
            # spectral convergence loss
            loss_sc = torch.norm(ref_mag - est_mag, p="fro") / torch.norm(
                ref_mag, p="fro"
            )
        elif reduction == "batch":
            loss_mag = torch.mean(
                self.loss_m(torch.log(ref_mag), torch.log(est_mag)), dim=[1, 2]
            )
            loss_sc = torch.norm(ref_mag - est_mag, p="fro", dim=[1, 2]) / torch.norm(
                ref_mag, p="fro", dim=[1, 2]
            )
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(
        self,
        fft_lens=(512, 1024, 2048),
        win_incs=(50, 120, 240),
        win_lens=(240, 600, 1200),
    ):
        super().__init__()
        self.loss_funcs = nn.ModuleList()
        for fft_len, win_inc, win_len in zip(fft_lens, win_incs, win_lens):
            self.loss_funcs.append(STFTLoss(fft_len, win_inc, win_len))

    def forward(self, ref_wav, est_wav, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(ref_wav, est_wav, reduction)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= len(self.loss_funcs)
        loss_mag /= len(self.loss_funcs)
        return loss_mag, loss_sc


def G_adv_loss(scores_fake, loss_func, reduction="mean"):
    """Compute G adversarial loss function and normalize values"""
    assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
    adv_loss = 0
    if isinstance(scores_fake, list):
        for score_fake in scores_fake:
            loss_fake = loss_func(score_fake, reduction)
            adv_loss += loss_fake
        adv_loss /= len(scores_fake)
    else:
        loss_fake = loss_func(scores_fake, reduction)
        adv_loss = loss_fake
    return adv_loss


def D_loss(scores_fake, scores_real, loss_func, reduction="mean"):
    """Compute D loss func and normalize loss values"""
    assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
    total_loss = 0
    total_loss_fake = 0
    total_loss_real = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            loss, loss_fake, loss_real = loss_func(score_fake, score_real, reduction)
            total_loss += loss
            total_loss_fake += loss_fake
            total_loss_real += loss_real
        # normalize loss values with number of scales (discriminators)
        total_loss /= len(scores_fake)
        total_loss_real /= len(scores_real)
        total_loss_fake /= len(scores_fake)
    else:
        # single scale loss
        total_loss, total_loss_real, total_loss_fake = loss_func(
            scores_fake, scores_real, reduction
        )
    return total_loss, total_loss_real, total_loss_fake


def D_loss_fake(scores_fake, loss_func, reduction="mean"):
    assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
    total_loss_fake = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake in scores_fake:
            loss_fake = loss_func._fake(score_fake, reduction)
            total_loss_fake += loss_fake
        # normalize loss values with number of scales (discriminators)
        total_loss_fake /= len(scores_fake)
    else:
        # single scale loss
        total_loss_fake = loss_func._fake(scores_fake, reduction)
    return total_loss_fake


def D_loss_real(scores_real, loss_func, reduction="mean"):
    assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
    total_loss_real = 0
    if isinstance(scores_real, list):
        # multi-scale loss
        for score_real in scores_real:
            loss_real = loss_func._real(score_real, reduction)
            total_loss_real += loss_real
        # normalize loss values with number of scales (discriminators)
        total_loss_real /= len(scores_real)
    else:
        # single scale loss
        total_loss_real = loss_func._real(scores_real, reduction)
    return total_loss_real


class MSEGLoss(nn.Module):
    """Mean Squared Generator Loss"""

    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss(reduction="none")

    def forward(self, score_fake, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_fake = self.loss_func(score_fake, score_fake.new_ones(score_fake.shape))
        if reduction == "mean":
            loss_fake = torch.mean(loss_fake)
        elif reduction == "batch":
            loss_fake = torch.mean(loss_fake, dim=1)
        return loss_fake


class HingeGLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_fake, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        if reduction == "mean":
            loss_fake = torch.mean(F.relu(1.0 - score_fake))
        elif reduction == "batch":
            loss_fake = torch.mean(F.relu(1.0 - score_fake), dim=1)
        return loss_fake


class MSEDLoss(nn.Module):
    """Mean Squared Discriminator Loss"""

    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss(reduction="none")

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_fake = self._fake(score_fake, reduction)
        loss_real = self._real(score_real, reduction)
        loss_d = loss_real + loss_fake
        return loss_d, loss_fake, loss_real

    def _fake(self, score_fake, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        if reduction == "mean":
            loss_fake = torch.mean(loss_fake)
        elif reduction == "batch":
            loss_fake = torch.mean(loss_fake, dim=1)
        return loss_fake

    def _real(self, score_real, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        if reduction == "mean":
            loss_real = torch.mean(loss_real)
        elif reduction == "batch":
            loss_real = torch.mean(loss_real, dim=1)
        return loss_real


class HingeDLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_fake = self._fake(score_fake, reduction)
        loss_real = self._real(score_real, reduction)
        loss_d = loss_real + loss_fake
        return loss_d, loss_fake, loss_real

    def _fake(self, score_fake, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        if reduction == "mean":
            loss_fake = torch.mean(F.relu(1.0 + score_fake))
        elif reduction == "batch":
            loss_fake = torch.mean(F.relu(1.0 + score_fake), dim=1)
        return loss_fake

    def _real(self, score_real, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        if reduction == "mean":
            loss_real = torch.mean(F.relu(1.0 - score_real))
        elif reduction == "batch":
            loss_real = torch.mean(F.relu(1.0 - score_real), dim=1)
        return loss_real


class FeatureLoss(nn.Module):
    """Calculates the feature matching loss, which is a learned similarity metric measured by
    the difference in features of the discriminator between a ground truth sample and a generated
    sample (Larsen et al., 2016, Kumar et al., 2019)."""

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.L1Loss(reduction="none")

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats, reduction="mean"):
        assert reduction in ["mean", "batch"], "reduction can only be 'mean' or 'batch'"
        loss_feats = 0
        num_feats = 0
        for idx, _ in enumerate(fake_feats):
            for fake_feat, real_feat in zip(fake_feats[idx], real_feats[idx]):
                loss_feat = self.loss_func(fake_feat, real_feat)
                if reduction == "mean":
                    loss_feat = torch.mean(loss_feat)
                elif reduction == "batch":
                    loss_feat = torch.mean(loss_feat, dim=[*(range(1, loss_feat.ndim))])
                loss_feats += loss_feat
                num_feats += 1
        loss_feats = loss_feats / num_feats
        return loss_feats


class SEANetGeneratorLoss(nn.Module):
    def __init__(
        self,
        hinge_g_loss=HingeGLoss(),
        hinge_g_loss_weight=1,
        feat_match_loss=FeatureLoss(),
        feat_match_loss_weight=100,
    ):
        super().__init__()
        self.hinge_g_loss = hinge_g_loss
        self.hinge_g_loss_weight = hinge_g_loss_weight
        self.feat_match_loss = feat_match_loss
        self.feat_match_loss_weight = feat_match_loss_weight

    def forward(self, scores_fake, feats_fake, feats_real, ref_wav=None, est_wav=None):
        adv_loss = 0
        return_dict = {}

        hinge_loss_fake = G_adv_loss(scores_fake, self.hinge_g_loss)
        return_dict["G_hinge_loss_fake"] = hinge_loss_fake
        adv_loss = adv_loss + self.hinge_g_loss_weight * hinge_loss_fake

        feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
        return_dict["G_feat_match_loss"] = feat_match_loss
        adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss

        return_dict["G_loss"] = adv_loss
        return_dict["G_adv_loss"] = adv_loss
        return return_dict


class SEANetDiscriminatorLoss(nn.Module):
    def __init__(self, hinge_d_loss=HingeDLoss()):
        super().__init__()
        self.hinge_d_loss = hinge_d_loss

    def forward(self, scores_fake, scores_real):
        loss = 0
        return_dict = {}

        hinge_D_loss, hinge_D_loss_real, hinge_D_loss_fake = D_loss(
            scores_fake, scores_real, self.hinge_d_loss
        )
        return_dict["D_hinge_gan_loss"] = hinge_D_loss
        return_dict["D_hinge_gan_loss_real"] = hinge_D_loss_real
        return_dict["D_hinge_gan_loss_fake"] = hinge_D_loss_fake
        loss += hinge_D_loss

        return_dict["D_loss"] = loss
        return return_dict


class AEROGeneratorLoss(nn.Module):
    def __init__(
        self,
        stft_loss=MultiScaleSTFTLoss(),
        stft_loss_weight=0.5,
        hinge_g_loss=HingeGLoss(),
        hinge_g_loss_weight=1,
        feat_match_loss=FeatureLoss(),
        feat_match_loss_weight=100,
    ):
        super().__init__()
        self.stft_loss = stft_loss
        self.stft_loss_weight = stft_loss_weight
        self.hinge_g_loss = hinge_g_loss
        self.hinge_g_loss_weight = hinge_g_loss_weight
        self.feat_match_loss = feat_match_loss
        self.feat_match_loss_weight = feat_match_loss_weight

    def forward(self, scores_fake, feats_fake, feats_real, ref_wav, est_wav):
        recon_loss = 0
        adv_loss = 0
        return_dict = {}

        stft_loss_mg, stft_loss_sc = self.stft_loss(ref_wav, est_wav)
        return_dict["G_stft_loss_mg"] = stft_loss_mg
        return_dict["G_stft_loss_sc"] = stft_loss_sc
        recon_loss = recon_loss + self.stft_loss_weight * (stft_loss_mg + stft_loss_sc)

        hinge_loss_fake = G_adv_loss(scores_fake, self.hinge_g_loss)
        return_dict["G_hinge_loss_fake"] = hinge_loss_fake
        adv_loss = adv_loss + self.hinge_g_loss_weight * hinge_loss_fake

        feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
        return_dict["G_feat_match_loss"] = feat_match_loss
        adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss

        return_dict["G_loss"] = adv_loss + recon_loss
        return_dict["G_adv_loss"] = adv_loss
        return_dict["G_recon_loss"] = recon_loss
        return return_dict
