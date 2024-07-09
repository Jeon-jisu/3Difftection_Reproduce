import csv
import os
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm
import zipfile
import shutil

ARkitscense_url = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
)


def get_training_video_ids(csv_file):
    video_ids = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fold"] == "Training":
                video_ids.append(row["video_id"])
    return video_ids


def download_file(url, file_name, dst):
    os.makedirs(dst, exist_ok=True)
    filepath = os.path.join(dst, file_name)

    if not os.path.isfile(filepath):
        command = f"curl {url} -o {file_name}.tmp --fail"
        print(f"Downloading file {filepath}")
        try:
            subprocess.check_call(command, shell=True, cwd=dst)
        except Exception as error:
            print(f"Error downloading {url}, error: {error}")
            return False
        os.rename(os.path.join(dst, file_name + ".tmp"), filepath)
    else:
        print(f"WARNING: skipping download of existing file: {filepath}")
    return True


def download_data(video_id, download_dir, assets):
    split = "Training"
    dst_dir = os.path.join(download_dir, "raw", split, str(video_id))
    url_prefix = f"{ARkitscense_url}/raw/{split}/{video_id}" + "/{}"

    for asset in assets:
        file_name = f"{asset}.zip" if asset != "lowres_wide.traj" else asset
        url = url_prefix.format(file_name)
        download_file(url, file_name, dst_dir)

        if file_name.endswith(".zip"):
            print(f"Extracting {file_name}")
            with zipfile.ZipFile(os.path.join(dst_dir, file_name), "r") as zip_ref:
                zip_ref.extractall(dst_dir)
            os.remove(os.path.join(dst_dir, file_name))


def sample_images(video_id, download_dir, time_threshold=1.0):
    traj_file = os.path.join(
        download_dir, "raw", "Training", str(video_id), "lowres_wide.traj"
    )
    df = pd.read_csv(traj_file, sep=" ", header=None)

    timestamps = df[0].apply(lambda x: round(x, 3))  # Round to 3 decimal places
    time_range = timestamps.max() - timestamps.min()
    interval = time_range / 4

    sampled_images = []
    for i in range(4):
        start_time = timestamps.min() + i * interval
        end_time = start_time + interval

        segment_df = df[(timestamps >= start_time) & (timestamps < end_time)]

        if len(segment_df) < 2:
            continue

        first_sample = segment_df.sample(n=1)
        first_timestamp = round(first_sample[0].iloc[0], 3)

        valid_second_samples = segment_df[
            (timestamps > first_timestamp)
            & (timestamps <= first_timestamp + time_threshold)
        ]

        if len(valid_second_samples) > 0:
            second_sample = valid_second_samples.sample(n=1)
            second_timestamp = round(second_sample[0].iloc[0], 3)

            for ts in [first_timestamp, second_timestamp]:
                image_name = f"{video_id}_{ts:.3f}.png"
                sampled_images.append((video_id, image_name, ts))

    return sampled_images


def move_sampled_images(video_id, download_dir, sampled_images):
    src_dir = os.path.join(download_dir, "raw", "Training", str(video_id))
    extract_dir = os.path.join(src_dir, "extract")
    extract_intrinsics_dir = os.path.join(src_dir, "extract_intrinsics")
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(extract_intrinsics_dir, exist_ok=True)

    moved_images = []
    for _, image_name, ts in sampled_images:
        # Move PNG files
        src_path = os.path.join(src_dir, "lowres_wide", image_name)
        dst_path = os.path.join(extract_dir, image_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_images.append((video_id, image_name, ts))
            print(f"Moved sampled image: {image_name}")
        else:
            print(
                f"Warning: Sampled image not found: {image_name}, src path : {src_path}"
            )

        # Move pincam files
        pincam_name = image_name.replace(".png", ".pincam")
        src_path_pincam = os.path.join(src_dir, "lowres_wide_intrinsics", pincam_name)
        dst_path_pincam = os.path.join(extract_intrinsics_dir, pincam_name)
        if os.path.exists(src_path_pincam):
            shutil.move(src_path_pincam, dst_path_pincam)
            print(f"Moved sampled intrinsics file: {pincam_name}")
        else:
            print(
                f"Warning: Sampled intrinsics file not found: {pincam_name}, src path : {src_path_pincam}"
            )

    # Delete the entire lowres_wide directory
    lowres_wide_dir = os.path.join(src_dir, "lowres_wide")
    if os.path.exists(lowres_wide_dir):
        shutil.rmtree(lowres_wide_dir)
        print(f"Deleted directory: {lowres_wide_dir}")
    else:
        print(f"Warning: Directory not found: {lowres_wide_dir}")

    # Delete the entire lowres_wide_intrinsics directory
    lowres_wide_intrinsics_dir = os.path.join(src_dir, "lowres_wide_intrinsics")
    if os.path.exists(lowres_wide_intrinsics_dir):
        shutil.rmtree(lowres_wide_intrinsics_dir)
        print(f"Deleted directory: {lowres_wide_intrinsics_dir}")
    else:
        print(f"Warning: Directory not found: {lowres_wide_intrinsics_dir}")

    return moved_images


def main(args):
    video_ids = get_training_video_ids(args.video_id_csv)
    sampled_data = []

    for video_id in tqdm(video_ids, desc="Processing videos"):
        download_data(video_id, args.download_dir, args.raw_dataset_assets)
        sampled_images = sample_images(video_id, args.download_dir, args.time_threshold)

        if sampled_images:
            moved_images = move_sampled_images(
                video_id, args.download_dir, sampled_images
            )
            sampled_data.extend(moved_images)

        if len(sampled_data) >= 40000:
            break

    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "image_name", "timestamp"])
        writer.writerows(sampled_data[:40000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and sample ARKitScenes dataset"
    )
    parser.add_argument(
        "--video_id_csv",
        default="raw/raw_train_val_splits.csv",
        help="Path to the CSV file containing video IDs",
    )
    parser.add_argument(
        "--download_dir",
        default="{다운받을 경로}",
        help="Directory to download the dataset",
    )
    parser.add_argument(
        "--output_file",
        default="sampled_images.csv",
        help="Output CSV file for sampled images",
    )
    parser.add_argument(
        "--raw_dataset_assets",
        nargs="+",
        default=["lowres_wide.traj", "lowres_wide", "lowres_wide_intrinsics"],
        help="Assets to download from the raw dataset",
    )
    parser.add_argument(
        "--time_threshold",
        type=float,
        default=1.0,
        help="Maximum time difference between image pairs in seconds",
    )
    args = parser.parse_args()

    main(args)
