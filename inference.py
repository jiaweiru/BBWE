import torch
import torchaudio
from tqdm import tqdm
from torch.nn import functional as F
from speechbrain.pretrained.interfaces import Pretrained
from speechbrain.dataio.dataio import load_data_json

import models
from models import LSD, ViSQOL


class BandWideExtension(Pretrained):
    MODULES_NEEDED = ["model", "original_sr", "target_sr", "hop_len"]

    def sr_batch(self, lr_wav):
        """Super-resolution a batch of low-resolution waveforms."""

        lr_wav = lr_wav.to(self.device)
        lr_samples = lr_wav.shape[-1]
        p = self.hparams.hop_len // (self.hparams.target_sr // self.hparams.original_sr)
        padding_len = (p - (lr_samples % p)) % p
        lr_wav = F.pad(lr_wav, (0, padding_len), "constant")
        sr_wav = self.mods.bwe_model(lr_wav)
        sr_wav = sr_wav[..., : lr_samples * 2]

        return sr_wav

    def sr_file(self, filename, output_filename=None):
        """Super-resolution a wav file."""

        lr_wav, sr = torchaudio.load(filename)
        assert (
            sr == self.hparams.original_sr
        ), f"Expect sample rate to {self.hparams.origin_sr}"
        sr_wav = self.sr_batch(lr_wav[None, :])[0]

        if output_filename is not None:
            torchaudio.save(
                output_filename, sr_wav, self.hparams.target_sr, bits_per_sample=16
            )

        return sr_wav


class AEROInferencer:
    def __init__(
        self, ckpt_path, original_sr=8000, target_sr=16000, hop_length=64, device="cpu"
    ):
        self.original_sr = original_sr
        self.target_sr = target_sr
        self.device = device

        self.model = models.AERO(
            lr_sr=original_sr, hr_sr=target_sr, hop_length=hop_length
        )
        state = torch.load(ckpt_path)["models"]["generator"]["state"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def match_signal(self, signal, ref_len):
        sig_len = signal.shape[-1]
        if sig_len < ref_len:
            signal = F.pad(signal, (0, ref_len - sig_len))
        elif sig_len > ref_len:
            signal = signal[..., :ref_len]
        return signal

    def sr_batch(self, lr_wav):
        """Super-resolution a batch of low-resolution waveforms."""

        lr_wav = lr_wav.to(self.device)
        ref_len = lr_wav.shape[-1] * (self.target_sr / self.original_sr)
        sr_wav = self.model(lr_wav)
        sr_wav = self.match_signal(sr_wav, ref_len)

        return sr_wav

    def sr_file(self, filename, output_filename=None):
        """Super-resolution a wav file."""

        lr_wav, sr = torchaudio.load(filename)
        assert sr == self.original_sr, f"Expect sample rate to {self.original_sr}"
        sr_wav = self.sr_batch(lr_wav[None])[0]

        if output_filename is not None:
            torchaudio.save(
                output_filename, sr_wav.cpu(), self.target_sr, bits_per_sample=16
            )

        return sr_wav

    def evaluate(self, json_file, data_folder):
        """Verify that model performance is consistent with the paper."""
        lsd = []
        visqol = []
        lsd_metric = LSD()
        visqol_metric = ViSQOL()

        data = load_data_json(json_file, replacements={"data_root": data_folder})
        data_ids = list(data.keys())
        hr_path_list = [data[i]["hr_path"] for i in data_ids]
        lr_path_list = [data[i]["lr_path"] for i in data_ids]

        with tqdm(list(zip(hr_path_list, lr_path_list))) as t:
            for hr_path, lr_path in t:
                hr_wav, _ = torchaudio.load(hr_path)
                hr_wav = hr_wav.to(self.device)
                lr_wav, _ = torchaudio.load(lr_path)
                lr_wav = lr_wav.to(self.device)

                sr_wav = self.sr_batch(lr_wav[None])[0]

                lsd.append(float(lsd_metric(hr_wav, sr_wav).detach()))
                visqol.append(float(visqol_metric(hr_wav, sr_wav).detach()))

                t.set_postfix(lsd=sum(lsd) / len(lsd), visqol=sum(visqol) / len(visqol))

        print(
            f"Test completed, lsd:{sum(lsd) / len(lsd)}, visqol:{sum(visqol) / len(visqol)}"
        )


if __name__ == "__main__":
    # # SB style inference
    # inferencer = BandWideExtension.from_hparams(
    #     source="./pretrained",
    #     hparams_file="inference.yaml",
    #     savedir="./pretrained",
    # )
    # inferencer.sr_file("test_lr.wav", "test_sr.wav")

    # AERO inference
    inferencer = AEROInferencer("pretrained/aero-nfft=512-hl=64.th", device="cuda:3")
    inferencer.evaluate("json/vctk_valid_8_16_new.json", "/home/sturjw/Datasets/VCTK")
