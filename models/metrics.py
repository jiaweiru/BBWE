import tempfile

import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch import nn

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from pathlib import Path

EPSILON = torch.finfo(torch.float32).eps


class LSD(nn.Module):
    def __init__(
        self, fft_len=2048, win_inc=512, win_len=2048, win_fn=torch.hann_window
    ):
        super().__init__()
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = win_fn(self.win_len)

    def forward(self, ref_wavs, est_wavs, reduction="mean"):
        ref_specs = torch.stft(
            ref_wavs,
            self.fft_len,
            self.win_inc,
            window=self.window.to(ref_wavs.device),
            return_complex=True,
        )
        ref_mags = torch.abs(ref_specs)
        est_specs = torch.stft(
            est_wavs,
            self.fft_len,
            self.win_inc,
            window=self.window.to(ref_wavs.device),
            return_complex=True,
        )
        est_mags = torch.abs(est_specs)

        ref_log = torch.log10(ref_mags.square().clamp(EPSILON))
        est_log = torch.log10(est_mags.square().clamp(EPSILON))

        lsd = (ref_log - est_log).square().mean(dim=1).sqrt().mean(dim=1)

        if reduction == "mean":
            return torch.mean(lsd)
        elif reduction == "batch":
            return lsd


class ViSQOL(nn.Module):
    def __init__(self, mode="speech", sr=16000):
        super().__init__()
        if mode == "audio":
            target_sr = 48000
            use_speech_scoring = False
            svr_model_path = "libsvm_nu_svr_model.txt"
        elif mode == "speech":
            target_sr = 16000
            use_speech_scoring = True
            svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

        self.sr = sr
        self.target_sr = target_sr
        svr_model_path = str(
            Path(visqol_lib_py.__file__).parent.joinpath("model", svr_model_path)
        )

        self.visqol_manager = visqol_lib_py.VisqolManager()
        self.visqol_manager.Init(
            visqol_lib_py.FilePath(svr_model_path), use_speech_scoring, False, 60, True
        )

    def forward(self, ref_wavs, est_wavs, reduction="mean"):
        if self.sr != self.target_sr:
            ref_wavs = torchaudio.functional.resample(ref_wavs, self.sr, self.target_sr)
            est_wavs = torchaudio.functional.resample(est_wavs, self.sr, self.target_sr)
        visqols = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for curinx in range(ref_wavs.shape[0]):
                sf.write(
                    "{}/est_{:07d}.wav".format(tmpdirname, curinx),
                    est_wavs[curinx].detach().cpu().numpy(),
                    self.target_sr,
                )
                sf.write(
                    "{}/ref_{:07d}.wav".format(tmpdirname, curinx),
                    ref_wavs[curinx].detach().cpu().numpy(),
                    self.target_sr,
                )
                out = self.visqol_manager.Run(
                    visqol_lib_py.FilePath(
                        "{}/ref_{:07d}.wav".format(tmpdirname, curinx)
                    ),
                    visqol_lib_py.FilePath(
                        "{}/est_{:07d}.wav".format(tmpdirname, curinx)
                    ),
                )
                visqols.append(out.moslqo)

        visqols = torch.from_numpy(np.array(visqols))

        if reduction == "mean":
            return torch.mean(visqols)
        elif reduction == "batch":
            return visqols
