import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

match_path = "/Users/melvinkisam/Documents/Python workspace/notts-dissertation/badminton-rally-classification/dataset-creation/match-videos/match_1-sb.mp4"
flags_path = "/Users/melvinkisam/Documents/Python workspace/notts-dissertation/badminton-rally-classification/dataset-creation/tag-output/match_1-sb_key.csv"
output_folder = '/Users/melvinkisam/Documents/Python workspace/notts-dissertation/badminton-rally-classification/dataset-creation/labelled-data/match_1'

def split_video(match_path, flags_path, output_folder, by='rallies'):

    df = pd.read_csv(flags_path, dtype={'time': float, 'key': int})
    print("Input Flags:")
    print(df)

    initial_df = pd.DataFrame({'time': [0], 'key': ['999']})

    df = pd.concat([initial_df,df], ignore_index=True)

    numClips = len(df)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    video = VideoFileClip(match_path)
    
    print('Saving in:', output_folder)

    for idx in range(len(df) - 1):
        time1 = df.loc[idx, "time"]
        time2 = df.loc[idx + 1, "time"]
        flag1 = df.loc[idx, "key"]
        flag2 = df.loc[idx + 1, "key"]

        if time1 > time2:
            raise ValueError(f'Time in row {idx+1} is greater than that in the next: \n {time1} : {time2}')
        
        # if flag1 == flag2:
        #     raise ValueError(f'Flag in row {idx+1} is the same as in the next.')

    for idx in range(len(df) - 1):  # Exclude the last row since it has no end time
        start_time = df.loc[idx, "time"]
        end_time = df.loc[idx + 1, "time"]
        flag = df.loc[idx + 1, "key"]

        #Decide to clip
        should_clip = False
        if by == 'rallies':
            should_clip = True
            output_file = f"{output_folder}/clip_{(idx+1)}.mp4"
        elif by == 'matches' and flag == 1:
            should_clip = True
            output_file = f"{output_folder}/match_{(idx+2)/2}.mp4"
        
        if should_clip:
            clip = video.subclipped(start_time, end_time)
                    
            clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
            print(f"Clip {idx}/{numClips} saved as {'rally' if flag else 'Not Rally'}")

    # Close the video file
    video.close()
    print("Video splitting completed!")

    if by == 'rallies':
        flag_output = output_folder + '/labels.csv'
        df[['key']].to_csv(flag_output, index=False, header=False)
        print(f"Flags saved to {flag_output}")


if not os.path.exists(output_folder): #Overwrite prevention... stops accidental runs loosing progress.
    split_video(match_path, flags_path, output_folder, by='rallies')


