import cv2
import numpy as np

# Useful to block out scores to anonymise data
# Pass array of 4 corners, draws a white box there on every frame of the video processed.

inputMatch = '/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/match-videos/match_1.mp4'
outputMatch = '/Users/melvinkisam/Documents/python-workspace/notts-dissertation/badminton-rally-classification/dataset-creation/match-videos/match_1-sb.mp4'

points = [[15,15], # top left
          [15,75], # bottom left
          [200,75], # bottom right
          [200,15]] # top right


def get_first_frame(video):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        ret, frame = cap.read()
        if ret:
            cap.release()
            return frame
        else:
            print("Error: Could not read the first frame:", video)
            cap.release()

def drawBox(video,points):
    frame = get_first_frame(video).copy()
    
    # Define the polygon (parallelogram) using the corners
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 0))  # Black polygon on the mask

    # Apply the mask to keep only the parallelogram region
    #masked_image = cv2.bitwise_and(frame, mask)

    cv2.imwrite('Training/Matches/test.jpg', frame)

def process_video(points, video, output_video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for .mp4 output (avc1 instead of mp4v for better compatibility and compression)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Create VideoWriter object
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    checkPoints = max(1, total_frames // 10)
    print(f"Processing {total_frames} frames... Checkpoints every {checkPoints} frames.")

    frameCount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the polygon on each frame
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 0))  # Black polygon

        # Write the modified frame to the output video
        out.write(frame)

        frameCount += 1
        if frameCount % checkPoints == 0:
            progress = (frameCount / total_frames) * 100
            print(f"Processed {frameCount}/{total_frames} frames ({progress:.1f}%)")

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video}")


# Function to align points
# drawBox(inputMatch,points)

# Function to append mask to entire video
process_video(points,inputMatch,outputMatch)