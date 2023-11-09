import sys

import torchaudio
from speechbrain.pretrained.interfaces import Pretrained


class BandWideExtension(Pretrained):
    MODULES_NEEDED = ["model", "original_sr", "target_sr"]

    def sr_batch(self, lr_wav):
        """Super-resolution a batch of low-resolution waveforms."""

        lr_wav = lr_wav.to(self.device)
        sr_wav = self.mods.bwe_model(lr_wav)

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


if __name__ == "__main__":
    inferencer = BandWideExtension.from_hparams(
        source="./pretrained",
        hparams_file="seanet_inference.yaml",
        savedir="./pretrained",
    )
    inferencer.sr_file("lr_test.wav", "sr_test.wav")
