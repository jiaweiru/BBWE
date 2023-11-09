import shutil

import sox
import json
import logging
import torchaudio
import numpy as np
import soundfile as sf
import typing as tp
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from speechbrain.processing.speech_augmentation import Resample
from speechbrain.utils.data_utils import get_all_files, download_file

logger = logging.getLogger(__name__)

# 28 speakers for training, and two speakers for testing
TRAIN_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y"
TEST_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y"


def prepare_vltn(
    data_folder: Path,
    save_json_train: Path,
    save_json_valid: Path,
    save_json_test: Path,
    original_sr: int = 8000,
    target_sr: int = 16000,
    res_type: str = "sox",
):
    """Prepares the json files for the Valentini dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.

    Args:
        data_folder (Path): the path to the the Valentini dataset.
        save_json_train (Path): the path to the train data specification file.
        save_json_valid (Path): the path to the valid data specification file.
        save_json_test (Path): the path to the test data specification file.
        original_sr (int): resample from 48kHz to given low sample rate.
        target_sr (int): resample from 48kHz to given high sample rate.
        res_type (str, optional): resample method. Defaults to "sox".
    """

    # Checks if this phase is already done (if so, skips it)
    if check_files(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    train_folder = data_folder.joinpath("clean_trainset_28spk_wav")
    test_folder = data_folder.joinpath("clean_testset_wav")
    train_archive = data_folder.joinpath("clean_trainset_28spk_wav.zip")
    test_archive = data_folder.joinpath("clean_testset_wav.zip")

    for folder, archive, url in zip(
        [train_folder, test_folder],
        [train_archive, test_archive],
        [TRAIN_URL, TEST_URL],
    ):
        if not check_folders(folder):
            logger.info(f"No data found for {folder}. Checking for an archive file.")
            if not archive.is_file():
                logger.info(
                    f"No archive file found for {archive}. Downloading and unpacking."
                )
                download_file(url, archive)
                logger.info(f"Downloaded data for {archive} from {url}.")
            else:
                logger.info(f"Found an archive file for {archive}. Unpacking.")

            shutil.unpack_archive(archive, data_folder)
    # exit()
    extension = [".wav"]
    train_list = []
    test_list = []  # Stores all audio file paths for the dataset
    train_list.extend(get_all_files(train_folder, match_or=extension))
    train_list = [Path(file) for file in train_list]
    test_list.extend(get_all_files(test_folder, match_or=extension))
    test_list = [Path(file) for file in test_list]

    for rate in [original_sr, target_sr]:
        # print(rate)
        if (
            data_folder.joinpath(f"clean_trainset_28spk_wav" + f"_{rate}").exists()
            and data_folder.joinpath(f"clean_testset_wav" + f"_{rate}").exists()
        ):
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
                wav.parents[1]
                .joinpath(f"{wav.parent.name}" + f"_{rate}")
                .joinpath(wav.name),
                rate,
            )
            for wav in tqdm(train_list + test_list)
        )

    # Creating json files
    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")
    create_json(train_list, save_json_train, original_sr, target_sr)
    create_json(test_list, save_json_valid, original_sr, target_sr)
    create_json(test_list, save_json_test, original_sr, target_sr)


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
        relative_path = Path("{data_root}").joinpath(*path_parts[-2:])

        # Gets the speaker-id from the utterance-id
        spk_id = uttid.split("_")[0]

        # Creates an entry for the utterance
        json_dict[uttid] = {
            "hr_path": str(relative_path).replace(
                path_parts[-2], path_parts[-2] + f"_{target_sr}"
            )
            if target_sr != 48000
            else str(relative_path),
            "lr_path": str(relative_path).replace(
                path_parts[-2], path_parts[-2] + f"_{original_sr}"
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


def resample_sox(input_dir: Path, output_dir: Path, target_sr: int):
    wav, sr = sf.read(input_dir)
    if (wav.shape[0] % 48) != 0:
        # Prevents sample point shifts caused by downsampling
        wav = np.pad(
            wav, (0, 48 - (wav.shape[0] % 48)), mode="constant", constant_values=0.0
        )  # use 48 as a fill factor to support up to 48x downsampling
        sf.write(input_dir, wav, sr)

    # 'soxr_hq' is the default setting of soxr
    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tfm.build_file(input_filepath=str(input_dir), output_filepath=str(output_dir))


def resample_torchaudio(input_dir: Path, output_dir: Path, target_sr: int):
    wav, _ = sf.read(input_dir)
    if (wav.shape[0] % 48) != 0:
        # Prevents sample point shifts caused by downsampling
        wav = np.pad(
            wav, (0, 48 - (wav.shape[0] % 48)), mode="constant", constant_values=0.0
        )  # use 48 as a fill factor to support up to 48x downsampling
        sf.write(input_dir, wav, 48000)

    signal, sig_sr = torchaudio.load(input_dir)
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
    prepare_vltn(
        Path("datasets/data"),
        Path("json/train.json"),
        Path("json/valid.json"),
        Path("json/test.json"),
        8000,
        32000,
    )

    # resample_sox(
    #     Path("/home/sturjw/Code/BandWidthExtension/p232_001.wav"),
    #     Path("resampled_sox.wav"),
    #     8000,
    # )
    # resample_torchaudio(
    #     Path("/home/sturjw/Code/BandWidthExtension/p232_001.wav"),
    #     Path("resampled_ta.wav"),
    #     8000,
    # )
