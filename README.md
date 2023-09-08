# Capstone_Project


**Background** 

Audio signals are comprised of three main elements:

- **Amplitude:** Is the distance between the highest and lowest points of a wave. The greater the distance of the two points refers to the volume of an audio signal. The image below shows the difference between louder and softer audio signals.
  
![Amplitude](https://qph.cf2.quoracdn.net/main-qimg-3bc3e189310f65661d8af5277a3b9872-pjlq) 

Image provided by [Pramitha P Kamath](https://www.quora.com/profile/Pramitha-P-Kamath)

---


<<<<<<< HEAD
=======
- **Frequency:** Is the amount of waves over a period of time. The more waves in a given period of time, the higher the frequency (pitch). The image below shows the difference between higher and lower frequencies. 

>>>>>>> 96b2260ae76a90770bc5b74fcad221a931f501dc
![Frequency](https://cdn.britannica.com/83/194283-004-37696A2F.jpg)

Image provided by [Britannica](https://kids.britannica.com/students/assembly/view/223513)

---


- **Time:** Will be interpreted as beats per measure and length of track in seconds. 

---



## Project Plan

1. Collect audio data, as .wav files, of chordal instruments playing variations of Major and Minor chords. This will include triads and 7th/6th chords in root, 1st, 2nd, or 3rd position. (Have collected 357 unique audio files of each chord quality totaling 714 data points. Plan to collect or create more.)
2. Create a dataframe of the name of each audio file and its respected classifying chord quality {Major: 0, Minor: 1}. (Have created a dataframe of the data I already have.)
3. Explore the metedata of the audio files collected using TinyTag to see any noticeable differences that could interfere with the model. (Not Started)
4. Load in the audio files using librosa and ensure all files have identical features by converting them to mono, standardizing the sample rate, and resizing them. (Not Started)
5. Transform the audio files into spectrograms to use as a visual to feed into the model. (Not Started)
6. Create a CNN binary classification model to identify the chord quality. If working efficiently, create a multi classification model. (Not Started)
7. Create a streamlit web app that allows users to upload audio to determine the chord quality. (Not Started)





