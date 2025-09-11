# badminton-rally-classification

A system where users can upload full match videos and receive a compilation of segmented clips categorised by key events, such as rallies, shuttle changes, player breaks, and umpire interactions, while also removing dead time where no meaningful activity occurs.

## Dataset Creation Procedure
1. **Collect match data** (`.mp4` format)  
2. **Run** `blockScore.py` to pseudonymise players  
3. **Run** `tagVideo.py` to annotate data  
4. **Run** `generateClips.py` to generate clips and their labels  
5. **Run** `generateVideoInfo.py` to generate video information as a `.txt` file  

## Model Training
There are two versions, 1 and 2. Refer to version 2 for the latest approach.
**Run** `train_model2.py` to train model on training set
**Run** `test_model2.py` to perform inference on validation set

## Automated Clip Generation
**Run** `match_classification.py` to segment a full match to defined classes (rally, not rally, shuttle change, floor mopping, set break)
**Run** `match_classification_text.py` to classify a full match to defined classes and output a full match with predictions

