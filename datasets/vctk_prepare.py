import shutil
import random

import sox
import json
import logging
import torchaudio
import numpy as np
import soundfile as sf
import typing as tp
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files, download_file

logger = logging.getLogger(__name__)
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y"
# README and license_text files do not need to be downloaded.

# Dataset split follows NU-Wave, NU-Wave2, AERO ...
OMIT_SPECIFIED = ["p280", "p315"]
VALID_SPECIFIED = [
    "p347",
    "p351",
    "p360",
    "p361",
    "p362",
    "p363",
    "p364",
    "p374",
    "p376",
    "s5",
]
TEST_SPECIFIED = [
    "p347",
    "p351",
    "p360",
    "p361",
    "p362",
    "p363",
    "p364",
    "p374",
    "p376",
    "s5",
]


def prepare_vctk(
    data_folder: Path,
    save_json_train: Path,
    save_json_valid: Path,
    save_json_test: Path,
    mic_id: tp.List[str] = ["mic1"],
    split: tp.Union[tp.List[int], str] = "specified",
    original_sr: int = 8000,
    target_sr: int = 16000,
    res_type: str = "sox",
):
    """Prepares the json files for the VCTK dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.

    Args:
        data_folder (Path): the path to the the VCTK dataset.
        save_json_train (Path): the path to the train data specification file.
        save_json_valid (Path): the path to the valid data specification file.
        save_json_test (Path): the path to the test data specification file.
        mic_id (tp.List[str], optional): microphone's id(mic1, mic2).
            Defaults to ["mic1"].
        split (tp.Union[tp.List[int], str], optional): When a list is used for the
            split, it indicates that the dataset will be divided proportionally.
            Conversely, when a string is used, it denotes that the dataset will be
            split according to specified rules. Defaults to "specified".
        original_sr (int): resample from 48kHz to given low sample rate.
        target_sr (int): resample from 48kHz to given high sample rate.
        res_type (str, optional): resample method. Defaults to "sox".
    """

    # Checks if this phase is already done (if so, skips it)
    if check_files(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    wav_folder = data_folder.joinpath("wav48_silence_trimmed")
    wav_archive = data_folder.joinpath("VCTK-Corpus-0.92.zip")

    extension = [i + ".flac" for i in mic_id]  # The expected extension for audio files
    wav_list = []  # Stores all audio file paths for the dataset

    if not check_folders(wav_folder):
        logger.info(f"No data found for {wav_folder}. Checking for an archive file.")
        if not wav_archive.is_file():
            logger.info(
                f"No archive file found for {wav_archive}. Downloading and unpacking."
            )
            url = VCTK_URL
            download_file(url, wav_archive)
            logger.info(f"Downloaded data for {wav_archive} from {url}.")
        else:
            logger.info(f"Found an archive file for {wav_archive}. Unpacking.")

        shutil.unpack_archive(wav_archive, data_folder)

    # Collects all files matching the provided extension
    wav_list.extend(get_all_files(wav_folder, match_or=extension))
    wav_list = [Path(file) for file in wav_list]

    for rate in [original_sr, target_sr]:
        if data_folder.joinpath(f"wav48_silence_trimmed" + f"_{rate}").exists():
            continue
        if rate == 48000:
            logger.info(f"Raw sampling rate of 48000, no further resampling required.")
            continue
        assert res_type in ["sox", "torchaudio"]

        if res_type == "sox":
            resample_fn = resample_sox
        elif res_type == "torchaudio":
            resample_fn = resample_torchaudio

        # speed up
        logger.info(f"Resample to {rate}.")
        Parallel(n_jobs=15)(
            delayed(resample_fn)(
                wav,
                Path(
                    str(wav).replace(
                        "wav48_silence_trimmed", f"wav48_silence_trimmed_{rate}"
                    )
                ),
                rate,
            )
            for wav in tqdm(wav_list)
        )

    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")

    # Random or specified split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split)

    # Creating json files
    create_json(data_split["train"], save_json_train, original_sr, target_sr)
    create_json(data_split["valid"], save_json_valid, original_sr, target_sr)
    create_json(data_split["test"], save_json_test, original_sr, target_sr)


def create_json(
    wav_list: tp.List[Path],
    json_file: Path,
    original_sr: int,
    target_sr: int,
) -> None:
    """Creates the json file given a list of wav files.

    Args:
        wav_list (tp.List[Path]): the list of wav files.
        json_file (Path): the path to the output json file.
        original_sr (int): resample from 48kHz to given low sample rate.
        target_sr (int): resample from 48kHz to given high sample rate.
    """
    json_dict = {}

    # Processes all the wav files in the list
    for wav_file in tqdm(wav_list):
        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        duration = signal.shape[1] / sig_sr

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.parts
        uttid = wav_file.stem
        relative_path = Path("{data_root}").joinpath(*path_parts[-3:])

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        # Creates an entry for the utterance
        json_dict[uttid] = {
            "hr_path": str(relative_path).replace(
                path_parts[-3], path_parts[-3] + f"_{target_sr}"
            )
            if target_sr != 48000
            else str(relative_path),
            "lr_path": str(relative_path).replace(
                path_parts[-3], path_parts[-3] + f"_{original_sr}"
            ),
            "spk_id": spk_id,
            "length": duration,
            "segment": True if "train" in str(json_file) else False,
        }

    # Writes the dictionary to the json file
    json_dir = json_file.parent
    json_dir.mkdir(parents=True, exist_ok=True)

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)


def split_sets(
    utterance_list: tp.List[Path],
    split: tp.Union[tp.List[int], str],
    shuffle: bool = True,
) -> tp.Dict[str, tp.List[Path]]:
    """Divide the given audio data into training, validation (and test) sets.

    Args:
        utterance_list (tp.List[Path]): the list of audio file paths.
        split (tp.Union[tp.List[int], str]): The rules for splitting the dataset.
        shuffle (bool, optional): Whether to shuffle the dataset, not applicable under
            specified and default split. Defaults to True.

    Returns:
        tp.Dict[str, tp.List[Path]]: the partitioned dataset.
    """

    if split == "specified":
        data_split = {
            "valid": [
                utterance
                for utterance in utterance_list
                if utterance.parent.name in VALID_SPECIFIED
            ],
            "test": [
                utterance
                for utterance in utterance_list
                if utterance.parent.name in TEST_SPECIFIED
            ],
            "train": [
                utterance
                for utterance in utterance_list
                if utterance.parent.name
                not in VALID_SPECIFIED + TEST_SPECIFIED + OMIT_SPECIFIED
            ],
        }
        return data_split

    else:
        # Random shuffles the list
        if shuffle:
            random.shuffle(utterance_list)

    tot_split = sum(split)
    tot_snts = len(utterance_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, set in enumerate(splits):
        n_snts = int(tot_snts * split[i] / tot_split)
        data_split[set] = utterance_list[0:n_snts]
        del utterance_list[0:n_snts]
    data_split["test"] = utterance_list

    return data_split


def resample_sox(input_dir: Path, output_dir: Path, target_sr: int):
    # wav, sr = sf.read(input_dir)
    # if (wav.shape[0] % 48) != 0:
    #     # Prevents sample point shifts caused by downsampling
    #     wav = np.pad(
    #         wav, (0, 48 - (wav.shape[0] % 48)), mode="constant", constant_values=0.0
    #     )  # use 48 as a fill factor to support up to 48x downsampling
    #     sf.write(input_dir, wav, sr)

    wav, sr = torchaudio.load(input_dir)
    if (wav.shape[1] % 48) != 0:
        wav = F.pad(wav, (0, 48 - (wav.shape[1] % 48)), mode="constant", value=0.0)
        torchaudio.save(input_dir, wav, sr, bits_per_sample=16)

    # 'soxr_hq' is the default setting of soxr
    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tfm.build_file(input_filepath=str(input_dir), output_filepath=str(output_dir))


def resample_torchaudio(input_dir: Path, output_dir: Path, target_sr: int):
    # wav, _ = sf.read(input_dir)
    # if (wav.shape[0] % 48) != 0:
    #     # Prevents sample point shifts caused by downsampling
    #     wav = np.pad(
    #         wav, (0, 48 - (wav.shape[0] % 48)), mode="constant", constant_values=0.0
    #     )  # use 48 as a fill factor to support up to 48x downsampling
    #     sf.write(input_dir, wav, 48000)

    signal, sig_sr = torchaudio.load(input_dir)
    signal = F.pad(signal, (0, 48 - (signal.shape[1] % 48)), mode="constant", value=0.0)
    if sig_sr == target_sr:
        return
    resampler = Resample(orig_freq=sig_sr, new_freq=target_sr)
    resampled_signal = resampler(signal)
    torchaudio.save(
        output_dir, resampled_signal, sample_rate=target_sr, bits_per_sample=16
    )


def check_files(*filenames: Path) -> bool:
    """Detects if the files exist.

    Returns:
        bool: return False if any of the files does not exist.
    """
    for filename in filenames:
        if not filename.is_file():
            return False
    return True


def check_folders(*folders: Path) -> bool:
    """Detects if the folders exist.

    Returns:
        bool: return False if any of the folders does not exist.
    """
    for folder in folders:
        if not folder.is_dir():
            return False
    return True


if __name__ == "__main__":
    prepare_vctk(
        Path("datasets/data"),
        Path("json/train.json"),
        Path("json/valid.json"),
        Path("json/test.json"),
        original_sr=8000,
        target_sr=32000,
    )
