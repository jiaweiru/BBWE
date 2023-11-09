import sys
import torch
import librosa
import torchaudio
import numpy as np
import speechbrain as sb
import matplotlib.pyplot as plt

from pathlib import Path
from pesq import pesq
from torch.nn import functional as F
from datasets import prepare_dataset
from speechbrain.nnet.loss import si_snr_loss
from hyperpyyaml import load_hyperpyyaml

from models import snr_loss

plt.switch_backend("agg")


# Brain class for bandwidth extension training
class BWEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Compute for loss calculation and inference in train stage and valid/test stage.

        Args:
            batch (PaddedBatch): This batch object contains all the relevant tensors for computation.
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns:
            sr_wav: estimated waveform.
        """

        batch = batch.to(self.device)
        lr_wav, _ = batch.lr_wav
        hr_wav, _ = batch.hr_wav
        # Mono and stereo compatible
        if len(lr_wav.shape) == 2:
            lr_wav = lr_wav.unsqueeze(1)
        if len(hr_wav.shape) == 2:
            hr_wav = hr_wav.unsqueeze(1)

        # Padding
        # This padding method applies to integer multiples of BWE
        lr_samples = lr_wav.shape[-1]
        hr_samples = hr_wav.shape[-1]
        p = self.hparams.hop_len // (self.hparams.target_sr // self.hparams.original_sr)
        padding_len = (p - (lr_samples % p)) % p
        lr_wav = F.pad(lr_wav, (0, padding_len), "constant")

        # model forward
        sr_wav = self.modules.generator(lr_wav)

        # Trimming
        sr_wav = sr_wav[:, :, :hr_samples]

        scores_fake, feats_fake = self.modules.discriminator(sr_wav.detach())
        scores_real, feats_real = self.modules.discriminator(hr_wav)

        return (sr_wav, scores_fake, feats_fake, scores_real, feats_real)

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given the predicted and targeted outputs.

        Args:
            predictions (torch.tensor): The output from `compute_forward`.
            batch (PaddedBatch): This batch object contains all the relevant tensors for computation.
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns:
            torch.Tensor: A one-element tensor used for backpropagating the gradient.
        """

        # Prepare clean targets for comparison
        sr_wav, scores_fake, feats_fake, scores_real, feats_real = predictions
        sr_wav = torch.squeeze(sr_wav, dim=1)
        hr_wav, lens = batch.hr_wav
        lr_wav, lens = batch.lr_wav

        loss_g = self.hparams.generator_loss(
            scores_fake, feats_fake, feats_real, hr_wav, sr_wav
        )
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d}

        if stage == sb.Stage.TEST and self.step <= self.hparams.log_save:
            self.hparams.tensorboard_train_logger.log_audio(
                f"{batch.id[0]}_ref", hr_wav, self.hparams.target_sr
            )
            self.hparams.tensorboard_train_logger.log_audio(
                f"{batch.id[0]}_sr", sr_wav, self.hparams.target_sr
            )

            est_mag, _ = librosa.magphase(
                librosa.stft(
                    sr_wav.cpu().detach().numpy(),
                    n_fft=2048,
                    hop_length=512,
                    win_length=2048,
                )
            )
            ref_mag, _ = librosa.magphase(
                librosa.stft(
                    hr_wav.cpu().detach().numpy(),
                    n_fft=2048,
                    hop_length=512,
                    win_length=2048,
                )
            )

            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            librosa.display.specshow(
                librosa.amplitude_to_db(np.squeeze(est_mag, axis=0)),
                cmap="magma",
                y_axis="linear",
                ax=axes[0],
                sr=self.hparams.target_sr,
            )
            axes[0].set_title("sr spec")
            librosa.display.specshow(
                librosa.amplitude_to_db(np.squeeze(ref_mag, axis=0)),
                cmap="magma",
                y_axis="linear",
                ax=axes[1],
                sr=self.hparams.target_sr,
            )
            axes[1].set_title("ref spec")
            plt.tight_layout()
            self.hparams.tensorboard_train_logger.writer.add_figure(
                f"{batch.id[0]}_Spectrogram", fig
            )

            Path.mkdir(Path(self.hparams.samples_folder), exist_ok=True)
            torchaudio.save(
                Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_ref.wav"),
                hr_wav.cpu(),
                self.hparams.target_sr,
                bits_per_sample=16,
            )
            torchaudio.save(
                Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_sr.wav"),
                sr_wav.cpu(),
                self.hparams.target_sr,
                bits_per_sample=16,
            )
            torchaudio.save(
                Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_lr.wav"),
                lr_wav.cpu(),
                self.hparams.original_sr,
                bits_per_sample=16,
            )

        self.sisnr_metric.append(batch.id, sr_wav, hr_wav, lens, reduction="batch")
        self.snr_metric.append(batch.id, sr_wav, hr_wav, lens, reduction="batch")
        self.lsd_metric.append(batch.id, hr_wav, sr_wav, reduction="batch")
        self.g_adv_metric.append(batch.id, scores_fake, reduction="batch")
        self.d_adv_real_metric.append(batch.id, scores_real, reduction="batch")
        self.d_adv_fake_metric.append(batch.id, scores_fake, reduction="batch")
        self.fm_metric.append(batch.id, feats_fake, feats_real, reduction="batch")

        if stage == sb.Stage.TEST:
            self.pesq_metric.append(batch.id, predict=sr_wav, target=hr_wav)
            self.visqol_metric.append(batch.id, hr_wav, sr_wav, reduction="batch")
        return loss

    def fit_batch(self, batch):
        """
        Train discriminator and generator adversarially
        """
        batch = batch.to(self.device)
        hr_wav, _ = batch.hr_wav
        if len(hr_wav.shape) == 2:
            hr_wav = hr_wav.unsqueeze(1)
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        sr_wav, scores_fake, feats_fake, scores_real, feats_real = outputs

        # First train the discriminator
        loss_d = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)["D_loss"]
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # Update adv value for generator training
        scores_fake, feats_fake = self.modules.discriminator(sr_wav)
        scores_real, feats_real = self.modules.discriminator(hr_wav)
        outputs = (sr_wav, scores_fake, feats_fake, scores_real, feats_real)

        # Then train the generator
        loss_g = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)["G_loss"]
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch"""
        outputs = self.compute_forward(batch, stage=stage)
        loss_g = self.compute_objectives(outputs, batch, stage=stage)["G_loss"]
        return loss_g.detach().cpu()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            opt_g_class, opt_d_class = self.opt_class
            # opt_g_class, opt_d_class, sch_g_class, sch_d_class = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(self.modules.discriminator.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer_g", self.optimizer_g)
                self.checkpointer.add_recoverable("optimizer_d", self.optimizer_d)

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch.

        Args:
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage. Defaults to None.
        """

        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.cpu().numpy(),
                deg=pred_wav.cpu().numpy(),
                mode="wb",
            )

        self.sisnr_metric = sb.utils.metric_stats.MetricStats(
            metric=si_snr_loss.si_snr_loss
        )
        self.snr_metric = sb.utils.metric_stats.MetricStats(metric=snr_loss)
        self.lsd_metric = sb.utils.metric_stats.MetricStats(metric=self.hparams.lsd)
        self.g_adv_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.g_adv_loss
        )
        self.d_adv_real_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.d_adv_loss_real
        )
        self.d_adv_fake_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.d_adv_loss_fake
        )
        self.fm_metric = sb.utils.metric_stats.MetricStats(metric=self.hparams.fm_loss)
        if stage == sb.Stage.TEST:
            self.pesq_metric = sb.utils.metric_stats.MetricStats(
                metric=pesq_eval, n_jobs=self.hparams.pesq_n_jobs, batch_eval=False
            )
            self.visqol_metric = sb.utils.metric_stats.MetricStats(
                metric=self.hparams.visqol
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Gets called at the end of an epoch.

        Args:
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
            stage_loss (float): The average loss for all of the data processed in this stage.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage. Defaults to None.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            # Define the train's stats as attributes to be counted at the valid stage.
            self.train_stats = {
                "sisnr": -self.sisnr_metric.summarize("average"),
                "snr": -self.snr_metric.summarize("average"),
                "lsd": self.lsd_metric.summarize("average"),
                "loss_generator_adv": self.g_adv_metric.summarize("average"),
                "loss_discriminator_adv_real": self.d_adv_real_metric.summarize(
                    "average"
                ),
                "loss_discriminator_adv_fake": self.d_adv_fake_metric.summarize(
                    "average"
                ),
                "loss_feature_match": self.fm_metric.summarize("average"),
            }
        # Summarize the statistics from the stage for record-keeping.

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            valid_stats = {
                "sisnr": -self.sisnr_metric.summarize("average"),
                "snr": -self.snr_metric.summarize("average"),
                "lsd": self.lsd_metric.summarize("average"),
                "loss_generator_adv": self.g_adv_metric.summarize("average"),
                "loss_discriminator_adv_real": self.d_adv_real_metric.summarize(
                    "average"
                ),
                "loss_discriminator_adv_fake": self.d_adv_fake_metric.summarize(
                    "average"
                ),
                "loss_feature_match": self.fm_metric.summarize("average"),
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            self.hparams.tensorboard_train_logger.log_stats(
                {"Epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best sisnr score.

            self.checkpointer.save_and_keep_only(
                meta={"epoch": epoch, **valid_stats},
                end_of_epoch=True,
                ckpt_predicate=(
                    lambda ckpt: (
                        ckpt.meta["epoch"] % self.hparams.keep_checkpoint_interval != 0
                    )
                )
                if self.hparams.keep_checkpoint_interval is not None
                else None,
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            test_stats = {
                "sisnr": -self.sisnr_metric.summarize("average"),
                "snr": -self.snr_metric.summarize("average"),
                "pesq": self.pesq_metric.summarize("average"),
                "lsd": self.lsd_metric.summarize("average"),
                "visqol": self.visqol_metric.summarize("average"),
            }

            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )


def dataio_prep(hparams):
    """
    This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Args:
        hparams (dict): This dictionary is loaded from the `train.yaml` file, and it includes all the hyperparameters needed for dataset construction and loading.

    Returns:
        dict: Contains two keys, "train" and "valid" that correspond to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline.
    # It is scalable for other requirements in coding tasks, such as enhancement, packet loss.
    @sb.utils.data_pipeline.takes("hr_path", "lr_path", "length", "segment")
    # Takes the key of the json file.
    @sb.utils.data_pipeline.provides("hr_wav", "lr_wav")
    # Provides("wav") -> using batch.wav
    def audio_pipeline(hr_path, lr_path, length, segment):
        """
        On the train stage, the audio is cropped and cut to a single channel in a given way.
        On the valid or test stage, the whole segment of the mixture audio is used, which is converted to a single channel using avg-to-mono.
        """
        if segment:
            segment_length = hparams["segment_length"]
            length = int(length * hparams["original_sr"])

            if length > segment_length:
                max_audio_start = length - segment_length
                audio_start = torch.randint(0, max_audio_start, (1,))
                audio_stop = audio_start + segment_length
                # Get wav with the shape [samples, channels]
                hr_wav = sb.dataio.dataio.read_audio(
                    {
                        "file": hr_path,
                        "start": audio_start
                        * (hparams["target_sr"] // hparams["original_sr"]),
                        "stop": audio_stop
                        * (hparams["target_sr"] // hparams["original_sr"]),
                    }
                )
                yield hr_wav

                lr_wav = sb.dataio.dataio.read_audio(
                    {
                        "file": lr_path,
                        "start": audio_start,
                        "stop": audio_stop,
                    }
                )
                yield lr_wav
            else:
                hr_wav = sb.dataio.dataio.read_audio(hr_path)
                lr_wav = sb.dataio.dataio.read_audio(lr_path)
                hr_wav = F.pad(hr_wav, (0, (segment_length - length) * 2), "constant")
                yield hr_wav
                lr_wav = F.pad(lr_wav, (0, (segment_length - length)), "constant")
                yield lr_wav

        else:
            hr_wav = sb.dataio.dataio.read_audio(hr_path)
            yield hr_wav

            lr_wav = sb.dataio.dataio.read_audio(lr_path)
            yield lr_wav

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "hr_wav", "lr_wav"],
            # "id" is implicitly added as an item in the data point.
        )
    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_dataset,
        kwargs={
            "corpus": hparams["corpus"],
            "data_folder": Path(hparams["data_folder"]),
            "save_json_train": Path(hparams["train_annotation"]),
            "save_json_valid": Path(hparams["valid_annotation"]),
            "save_json_test": Path(hparams["test_annotation"]),
            "original_sr": hparams["original_sr"],
            "target_sr": hparams["target_sr"],
        },
    )

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    nc_brain = BWEBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    nc_brain.fit(
        epoch_counter=nc_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Load best checkpoint (highest SISNR) for evaluation
    test_stats = nc_brain.evaluate(
        test_set=datasets["test"],
        # max_key="sisnr",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
