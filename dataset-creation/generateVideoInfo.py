import pandas as pd
import os

match_folder = 'match_9-sb'
labels_path = f'/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/dataset/labelled-data/{match_folder}'
output_path = f'/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/dataset/training-input/val'


def create_video_info(clips_dir: str, label_df: pd.DataFrame, output_dir: str):
    output_lines = []

    # Create a dict from clip name to label
    label_dict = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))

    for filename in sorted(os.listdir(clips_dir)):
        if filename.endswith('.mp4') and filename.startswith('clip_'):
            clip_path = os.path.join(clips_dir, filename)

            if filename in label_dict:
                label = int(label_dict[filename])
                output_lines.append(f'{clip_path} {label}')
            else:
                print(f"[WARNING] No label found for clip: {filename}")

    # Save to output file
    match_id = os.path.basename(clips_dir)
    output_file = os.path.join(output_dir, f'{match_id}.txt')

    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    print(f"[INFO] Saved {len(output_lines)} labeled entries to {output_file}")


if __name__ == '__main__':
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Folder not found: {labels_path}")

    labels_csv = os.path.join(labels_path, 'labels.csv')
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"'labels.csv' not found in {labels_path}")

    labels_df = pd.read_csv(labels_csv, header=None)
    print(f"[INFO] Loaded {len(labels_df)} label entries from labels.csv")

    create_video_info(labels_path, labels_df, output_path)
