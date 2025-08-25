# badminton-rally-classification

A system where users can upload full match videos and receive a compilation of segmented clips categorised by key events, such as rallies, shuttle changes, player breaks, and umpire interactions, while also removing dead time where no meaningful activity occurs.

## Dataset Creation Procedure
1. **Collect match data** (`.mp4` format)  
2. **Run** `blockScore.py` to pseudonymise players  
3. **Run** `tagVideo.py` to annotate data  
4. **Run** `generateClips.py` to generate clips and their labels  
5. **Run** `generateVideoInfo.py` to generate video information as a `.txt` file  

## Automated Clip Generation
**Run** `match_classification.py` to segment a full match to defined classes (rally, not rally, shuttle change, floor mopping, set break)