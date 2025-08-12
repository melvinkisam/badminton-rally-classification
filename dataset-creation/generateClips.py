import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

matchID = 'match_9-sb'
match_path = f"./badminton-rally-classification/dataset-creation/match-videos/{matchID}.mp4"
flags_path = f"./badminton-rally-classification/dataset-creation/tag-output/{matchID}_key.csv"
output_folder = f'./badminton-rally-classification/dataset-creation/dataset/labelled-data/{matchID}'


def split_video(match_path, flags_path, output_folder, by='rallies'):
    df = pd.read_csv(flags_path, header=0, dtype={'time': float, 'key': int})
    print("Input Flags:")
    print(df)

    num_clips = len(df)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    video = VideoFileClip(match_path)
    print('Saving in:', output_folder)

    label_entries = []

    for idx in range(len(df)):
        # Since the split needs a start and end time, we assume the first clip starts at 0
        if idx == 0:
            start_time = 0.0
        else:
            start_time = df.loc[idx - 1, "time"]

        end_time = df.loc[idx, "time"]
        label = df.loc[idx, "key"]

        if start_time >= end_time:
            raise ValueError(f"[ERROR] Start time >= end time at index {idx}: {start_time} >= {end_time}")

        output_filename = f"clip_{idx+1}.mp4"
        output_path = os.path.join(output_folder, output_filename)

        clip = video.subclip(start_time, end_time)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Clip {idx+1}/{num_clips} saved: {output_filename}")

        label_entries.append((output_filename, label))

    video.close()
    print("Video splitting completed!")

    if by == 'rallies':
        label_df = pd.DataFrame(label_entries, columns=["filename", "label"])
        label_csv_path = os.path.join(output_folder, "labels.csv")
        label_df.to_csv(label_csv_path, index=False)
        print(f"Labels saved to: {label_csv_path}")


if not os.path.exists(output_folder):
    split_video(match_path, flags_path, output_folder, by='rallies')
