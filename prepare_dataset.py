#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned & standardized script for PSG (EEG/EOG/EMG) chunking to epoch-level NPZ.

Pipeline:
- Read EDF
- Average reference
- Optional 50 Hz notch
- Optional band-pass (0.3–45 Hz)
- Optional resample to --df
- Epoch into 30 s segments
- Read labels from paired .xlsx via label2list()
- Map labels to numeric via ann2label
- Drop MOVE/UNK
- Channel-wise normalization
- Save as NPZ

Requirements:
- mne
- numpy
- Project utils:
    - prepare_datasets.Utils.sleep_state.stage_dict
    - prepare_datasets.Utils.logger.get_logger
    - prepare_datasets.Utils.preprocess.Channel_normalization
    - sleep_ays_script.Utils.labels_utils.label2list
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

# --- Project utils (kept as in your repo) ---
try:
    from prepare_datasets.Utils.sleep_state import stage_dict
    from prepare_datasets.Utils.logger import get_logger
    from prepare_datasets.Utils.preprocess import Channel_normalization
    from sleep_ays_script.Utils.labels_utils import label2list
except Exception as e:
    print(
        "[ERROR] Failed to import project utilities. "
        "Please ensure your PYTHONPATH includes the project root.\n"
        f"Original error: {e}",
        file=sys.stderr,
    )
    raise

try:
    import mne
except Exception as e:
    print("[ERROR] mne is required. Install via `pip install mne`.", file=sys.stderr)
    raise

# ---------------- Constants ---------------- #
EPOCH_SEC_SIZE = 30

ANN2LABEL = {
    "W": 0, "Wake": 0, "4": 0,
    "1": 1, "N1": 1,
    "2": 2, "N2": 2,
    "3": 3, "N3": 3,
    "R": 4, "5": 4, "REM": 4, "Rem": 4,
    "X": 5, "Unknown": 5, "0": 5, "Art": 5,
}

# ---------------- Helpers ---------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract epochs from PSG EDF and save as NPZ."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing EDF and corresponding XLSX annotation files "
             "(annotation file should share the same stem as EDF).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save NPZ outputs.",
    )
    p.add_argument(
        "--select-ch",
        type=str,
        default="F3,F4,C3,C4,O1,O2",
        help="Comma-separated EEG channel names to include (default: F3,F4,C3,C4,O1,O2).",
    )
    p.add_argument(
        "--apply-filter",
        action="store_true",
        help="Apply band-pass filter (0.3–45 Hz).",
    )
    p.add_argument(
        "--no-apply-filter",
        action="store_false",
        dest="apply_filter",
        help="Disable band-pass filter.",
    )
    p.set_defaults(apply_filter=True)

    p.add_argument(
        "--apply-notch",
        action="store_true",
        help="Apply 50 Hz notch filter.",
    )
    p.add_argument(
        "--no-apply-notch",
        action="store_false",
        dest="apply_notch",
        help="Disable 50 Hz notch filter.",
    )
    p.set_defaults(apply_notch=True)

    p.add_argument(
        "--df",
        type=int,
        default=100,
        help="Downsample frequency (Hz). Use 0 to disable resampling.",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="info_ch_extract.log",
        help="Log file name (will be created under --output-dir).",
    )
    return p.parse_args()


def safe_label_map(raw_labels: list) -> np.ndarray:
    """
    Map raw stage labels to numeric using ANN2LABEL. Unknown keys -> 'Unknown' (5).
    """
    mapped = []
    for lab in raw_labels:
        key = str(lab).strip()
        mapped.append(ANN2LABEL.get(key, ANN2LABEL["Unknown"]))
    return np.asarray(mapped, dtype=np.int64)


def epoch_count(n_times: int, sf: float, epoch_sec: int) -> int:
    return int(n_times // (epoch_sec * sf))


# ---------------- Main ---------------- #
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / args.log_file
    logger = get_logger(str(log_path), level="info")

    select_ch = [c.strip() for c in args.select_ch.split(",") if c.strip()]
    if not select_ch:
        logger.error("No channels provided via --select-ch.")
        sys.exit(1)

    edf_paths = sorted(args.data_dir.glob("*.edf"))
    if not edf_paths:
        logger.error(f"No EDF files found under: {args.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(edf_paths)} EDF files.")
    logger.info(f"Channels: {select_ch}")
    logger.info(f"Filters -> Notch(50Hz): {args.apply_notch} | Band-pass(0.3–45Hz): {args.apply_filter}")
    logger.info(f"Downsample to: {'no resample' if args.df == 0 else args.df} Hz")

    for edf_path in edf_paths:
        try:
            logger.info(f"Processing EDF: {edf_path.name}")
            # Read EDF with selected channels
            raw = mne.io.read_raw_edf(
                str(edf_path),
                include=select_ch,
                preload=True,
                verbose="ERROR",
            )

            orig_sf = float(raw.info["sfreq"])
            logger.info(f"Original sampling rate: {orig_sf} Hz")
            logger.info(f"Epoch duration: {EPOCH_SEC_SIZE} sec")

            # Average reference
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")

            # 50 Hz notch
            if args.apply_notch:
                raw.notch_filter(freqs=50, notch_widths=2, verbose="ERROR")

            # Band-pass 0.3–45 Hz
            if args.apply_filter:
                raw.filter(l_freq=0.3, h_freq=45.0, verbose="ERROR")

            # Optional resampling
            if args.df and args.df > 0 and args.df != int(orig_sf):
                raw.resample(sfreq=int(args.df), verbose="ERROR")
                sf = float(raw.info["sfreq"])
            else:
                sf = orig_sf
            logger.info(f"Working sampling rate: {sf} Hz")

            # Compute epoch count from working sampling rate
            n_epochs = epoch_count(raw.n_times, sf, EPOCH_SEC_SIZE)
            if n_epochs == 0:
                logger.warning("File too short for a single 30 s epoch; skipping.")
                continue

            # Get data (channels x samples)
            data = raw.get_data(picks=select_ch)  # shape: (n_ch, n_samples)
            n_epoch_samples = int(EPOCH_SEC_SIZE * sf)
            total_samples = n_epochs * n_epoch_samples
            data = data[:, :total_samples]
            # reshape -> (n_epochs, n_ch, n_epoch_samples)
            data = data.reshape(len(select_ch), n_epochs, n_epoch_samples).transpose(1, 0, 2)
            logger.info(f"Signal shape (epochs, ch, samples): {data.shape}")

            # --- Labels ---
            ann_xlsx = edf_path.with_suffix(".xlsx")
            if not ann_xlsx.exists():
                logger.error(f"Missing annotation file for {edf_path.name}: {ann_xlsx.name}")
                logger.error("Skipping this EDF.")
                continue

            logger.info(f"Annotation file: {ann_xlsx.name}")
            raw_stage_list = label2list(str(ann_xlsx), usecol=[2])  # expects a flat list/array
            labels = safe_label_map(raw_stage_list)

            # Align labels to epochs length
            if len(labels) < n_epochs:
                logger.warning(f"Labels shorter than signal epochs ({len(labels)} < {n_epochs}); truncating signal.")
                n_epochs = len(labels)
                total_samples = n_epochs * n_epoch_samples
                data = data[:n_epochs]
            elif len(labels) > n_epochs:
                logger.warning(f"Labels longer than signal epochs ({len(labels)} > {n_epochs}); truncating labels.")
                labels = labels[:n_epochs]

            # Sanity check
            if len(labels) != n_epochs:
                logger.error(f"Epoch/label mismatch after alignment: {len(labels)} vs {n_epochs}")
                continue

            # Remove MOVE & UNK
            move_idx = np.where(labels == stage_dict["MOVE"])[0]
            unk_idx = np.where(labels == stage_dict["UNK"])[0]
            if move_idx.size or unk_idx.size:
                remove_idx = np.union1d(move_idx, unk_idx)
                keep_idx = np.setdiff1d(np.arange(n_epochs), remove_idx)
                logger.info(
                    f"Removing MOVE({move_idx.size}) & UNK({unk_idx.size}); "
                    f"kept epochs: {keep_idx.size}/{n_epochs}"
                )
                data = data[keep_idx]
                labels = labels[keep_idx]
                n_epochs = data.shape[0]

            # Normalize (channel-wise)
            data = Channel_normalization(data)

            # Save
            out_name = edf_path.stem + ".npz"
            out_path = args.output_dir / out_name
            np.savez(
                out_path,
                x=data.astype(np.float64),
                y=labels.astype(np.int64),
                fs=sf,
                ch_label=np.asarray(select_ch, dtype="U"),
                epoch_duration=EPOCH_SEC_SIZE,
                n_all_epochs=int(epoch_count(raw.n_times, sf, EPOCH_SEC_SIZE)),
                n_epochs=int(n_epochs),
            )
            logger.info(f"Saved: {out_path}")
            logger.info("=" * 40 + "\n")

        except Exception as e:
            logger.exception(f"Failed on file: {edf_path.name} | Error: {e}")


if __name__ == "__main__":
    main()
