# badminton-rally-classification
A system where users can upload full match videos and receive a compilation of segmented clips categorised by key events, such as rallies, shuttle changes, player breaks, and umpire interactions, while also removing dead time where no meaningful activity occurs.

Dataset creation procedure:
STEP 1 - Collect match data (.mp4 format)
STEP 2 - Run blockScore.py to pseudonymise players
STEP 3 - Run tagVideo.py to annotate data
STEP 4 - Run generateClips.py to generate clips and their labels
STEP 5 - Run generateVideoInfo.py to generate video information as a .txt file