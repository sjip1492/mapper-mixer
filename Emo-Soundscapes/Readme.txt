Thank you for downloading Emo-Soundscapes.

What Emo-Soundscapes is:

Emo-Soundscapes is the Annotated Creative Commons Emotional Database. It has 1213 audio clips. It is composed of two subsets:
- 600 soundscapes recordings: 600 soundscapes clips are extracted from soundscape recordings in Freesound.org. 
- 613 mixed soundscapes recordings: 613 mixed sounds are created based on the 600 soundscapes recordings. Each mix consists of mixing two or three selected audio clips selected at within and between Schafer’s soundscape categories and modulating and attenuation level.

The 1213 soundscape clips (each 6 seconds long) are ranked along the perceived valence, from the soundscape recordings perceived the most negatively to the soundscape recordings perceived the most positively. The 1213 soundscape clips are ranked sorted along the perceived arousal axis. The annotation was carried out by 1182 annotators from 74 different countries using crowdsourcing. Because Freesound.org shared under Creative Commons licenses, it allows to make this database publicly available without copyright issues

How to use Emo-Soundscapes:

Emo-Soundscapes-Audio
_ /600_Sounds
    The 600 wav audio excerpts are saved in the data folder. We keep 100 excerpts for each Schafer’s category.
	
_ /613_MixedSounds            
    The 613 wav mixed audio excerpts are saved in the data folder. They are classified by the number of audio clips that are used to mix and the between/within Schafer’s category. 

_ /AudioClipsUsedForMix(A+B)
    These are the audio clips that are selected for mixing (A+B). They are classified by the level of attenuation. 

_ /AudioClipsUsedForMix(A+B+C)
    These are the audio clips that are selected for mixing (A+B+C). They are classified by the level of attenuation. 
 
Emo-Soundscapes-Features 
_ /Features.csv
    This file includes all the features (see description below). 

_ /Normalized_Features.csv
    This file includes all the normalized features. We normalize the audio features between 0 and 1.0.


Emo-Soundscapes-Metadata
    The metadata of 600 soundscapes recordings 
	FileName	
	SearchTerm: The term used for retrieval the sound  	
	Duration: Duration of the file 	
	Class: Background Sound/Foreground Sound/Background with Foreground Sound	
	FsID: Freesound ID 	
	FsUrl: Freesound URL 	
	Tags: Tags on Freesound.org 


Emo-Soundscapes-Rankings 
_ /Arousal.csv
	This file is composed of 2 columns:
	_ FileName: file name of the excerpt saved in /data
	_ Ranking: rank of the excerpt in the database along the arousal axis

_ /Valence.csv
	This file is composed of 2 columns:
	_ FileName: file name of the excerpt saved in /data
	_ Ranking: rank of the excerpt in the database along the valence axis

Emo-Soundscapes-Ratings 

Ratings are converted from rankings to train regression models. This procedure has two assumptions. First, the distances between two successive rankings are equal. Second, the valence and arousal are in the range of [-1.0, 1.0]. We map the range of ranking values, 1 to 1213, to a corresponding rating range of 1.0 to -1.0, respectively.

_ /Arousal.csv
	This file is composed of 2 columns:
	_ FileName: file name of the excerpt saved in /data
	_ Rating: rating of the excerpt in the database along the arousal axis

_ /Valence.csv
	This file is composed of 2 columns:
	_ FileName: file name of the excerpt saved in /data
	_ Rating: rating of the excerpt in the database along the valence axis


