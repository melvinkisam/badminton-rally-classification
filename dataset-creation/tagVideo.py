import pygame
import csv
from moviepy.editor import VideoFileClip

filename = "match_2-sb"
video_path = f"/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/match-videos/{filename}.mp4"
output_csv = f"/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/tag-output/{filename}_key.csv"

inputs = []
dimensions = [1000, 600]


def save_inputs_to_csv(inputs, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "key"])
        writer.writeheader()
        writer.writerows(inputs)

def play_video_with_inputs(video_path):
    pygame.init()
    screen = pygame.display.set_mode(dimensions)
    pygame.display.set_caption("Video Player")

    video = VideoFileClip(video_path).resize(dimensions)
    fps = video.fps
    frame_duration = 1 / fps
    playbackSpeed = 2
    frame_idx = 0
    current_time = 0.0
    paused = False
    running = True
    pending_label = None  # Store pending number key

    print("Controls:")
    print("  SPACE - pause/resume")
    print("  LEFT/RIGHT - adjust time by 0.5s (when paused)")
    print("  0-9 - select label key (auto-pauses)")
    print("  RETURN - save selected key")
    print("  BACKSPACE - abort label or reassign")
    print("  Q - quit and save")

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

                if key == 'q':
                    running = False
                    break

                elif key == 'space':
                    paused = not paused
                    if not paused:
                        # Sync frame index to new current_time after pause adjustments
                        frame_idx = int(current_time * fps)
                    print("Paused" if paused else "Resumed")

                elif key in ["left", "left arrow"] and paused:
                    current_time = max(0.0, current_time - 0.5)
                    print(f"Moved back to {round(current_time, 2)}s")

                elif key in ["right", "right arrow"] and paused:
                    current_time = min(video.duration, current_time + 0.5)
                    print(f"Moved forward to {round(current_time, 2)}s")

                elif key.isdigit() and len(key) == 1:
                    pending_label = key
                    paused = True
                    print(f"Key '{pending_label}' selected at time {round(current_time, 2)}s")

                elif key == "return" and pending_label is not None:
                    inputs.append({"time": round(current_time, 2), "key": pending_label})
                    print(f"Saved key '{pending_label}' at {round(current_time, 2)}s")
                    pending_label = None

                elif key == "backspace" and pending_label is not None:
                    print(f"Aborted key '{pending_label}'")
                    pending_label = None

        if not paused:
            frame = video.get_frame(current_time)
            pygame_frame = pygame.image.frombuffer(frame.tobytes(), video.size, "RGB")
            screen.blit(pygame_frame, (0, 0))
            pygame.display.flip()

            frame_idx += playbackSpeed
            current_time = frame_idx * frame_duration

            if current_time >= video.duration:
                break

            pygame.time.delay(int(1000 / (fps * playbackSpeed)))

        else:
            frame = video.get_frame(current_time)
            pygame_frame = pygame.image.frombuffer(frame.tobytes(), video.size, "RGB")
            screen.blit(pygame_frame, (0, 0))
            pygame.display.flip()
            clock.tick(30)

    save_inputs_to_csv(inputs, output_csv)
    print(f"Inputs saved to {output_csv}")
    pygame.quit()


play_video_with_inputs(video_path)
print("Program finished.")
