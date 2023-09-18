# Audio Classification Project 

---

## Problem Statement 

Convolutional Neural Network (CNN) models are primarily used for image classification, therefore raw audio files will not work well as input data. We are looking to build a CNN model using TensorFlow that can classify the difference between a Major and Minor chord, recorded from a piano or guitar, by processing the audio files as spectrogram images. Using spectrograms is the most common current method for audio classification. Although there has been success with using these graphs for classification models, spectrograms inherently aren't images and need augmentations.

This project aims to determine: 

- Which formated spectrogram performs best for a CNN model? 

- Which model performs best to determine audio binary classification? 

In this project we will test different augmented spectrograms on CNN models and determine our success based on how it compares to our **null_baseline of 58.4% accuracy.** This project can give insight to what models and which spectrograms should be used for future audio classification projects.    

---

## Folder-Path Description 

**audio_files**: contains all of the .wav raw audio files from the major and minor kaggle dataset

**data**: contains all of the spectrogram images saved from the 02_preprocessing notebook

**images**: used to store every image saved to use for the presentation

**code**: has the 01_eda_audio, 02_preprocessing, 03_final_models notebooks which is all of the code needed for this project

**audio_files.csv**: dataframe created from the 01_eda_audio that contains every audio file path name and is target variable 

**presentation.pdf**: a slide-show summarizing the project


---

## Introduction 

In music there are four main categories of chord qualities: Major, Minor, Diminished, and Augmented. Each of these chords have a specific notation shape that determines its class, however the order of the notation can vary incredibly. For this reason in this project we will be exclusively focusing on Major and Minor chords. The dataset used for this project contains a large variety of notation orders for both of these chord qualities. The terminology below are important concepts to understand before engaging in this project.


**Background and Terminology** 


- **Amplitude:** Is the distance between the highest and lowest points of a wave. The greater the distance of the two points refers to the volume of an audio signal. The image below shows the difference between louder and softer audio signals.
  
![Amplitude](https://qph.cf2.quoracdn.net/main-qimg-3bc3e189310f65661d8af5277a3b9872-pjlq) 

Image provided by [Pramitha P Kamath](https://www.quora.com/profile/Pramitha-P-Kamath)

---


- **Frequency:** Is the amount of waves over a period of time. The more waves in a given period of time, the higher the frequency (pitch). The image below shows the difference between higher and lower frequencies. 

![Frequency](https://cdn.britannica.com/83/194283-004-37696A2F.jpg)

Image provided by [Britannica](https://kids.britannica.com/students/assembly/view/223513)

---


- **Mel/Spectrogram:** A spectrogram is a visual way to represent the amplitude of frequencies over time. The color of the frequency in a spectrogram determines its amplitude and the x-axis represents frequency. Spectrograms are a two-dimensional graph with the third dimension (amplitude) represented by color. A Mel-Spectrogram uses a fourier transformer to convert the frequencies to the mel_frequency scale. This can help the resolution with lower frequencies that otherwise would be harder to visually represent. 

![Spectrogram](https://i0.wp.com/www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2023/03/spectrogram-of-audio-example.png?w=564&ssl=1)

Image provided by [GaussianWaves](https://www.gaussianwaves.com/2022/03/spectrogram-analysis-using-python/)

---


- **Inversions**: Fundamentally a chord is constructed out of the first, third, fifth, and seventh note of a seven note scale. Inversions are different ways to order a chord. Here are the most basic examples of chord inversions.

- Root Position = 1, 3, 5, 7
- 1rst Inversion = 3, 5, 7, 1
- 2nd Inversion = 5, 7, 1, 3
- 3rd Inversion = 7, 1, 3, 5

---

- **Extensions**: Are added notes to chords that are not the fundamental 1st, 3rd, 5th, and 7th notes of the seven note scale. Most common extensions are 9th (2nd note of scale up an octave) and 11th(4th note up an octave).

An example would be a F Minor 9th chord.

- Notes of Scale: 1_F, 2_G, 3_Ab, 4_Bb, 5_C, 6_D, 7_Eb
- Notes of Chord: F, Ab, C, Eb, G
- Numeric Quality of Notes: 1, 3, 5, 7, 9

---

## Notable Tools 

- Librosa: this is a python package used for audio extraction and analysis
- IPython: used to play and listen to audio files from notebook
- Pillow: this is a python image package used for image extraction and analysis 
- Keras: is a Python API used to work with neural networks 
- TensorFlow: a large encompassing machine learning library 
- Sci-kit Learn: used for statistical modeling applications 
- VGG16:  A transfer learning pretrained model specified in image classification 


## Dataset 

The dataset used for this project was taken from Kaggle and created by Jaipur, Rajasthan. 
[Major_v_Minor_Data](https://www.kaggle.com/datasets/deepcontractor/musical-instrument-chord-classification) 
This data consists of two classes, Major and Minor, that are divided into two folders. Every audio file in the dataset is in the .wav file format. Rajasthan also mentions in his Kaggle description that the audio files are recordings of guitar and piano scraped from various sources. 

We will create a dataframe composed of the file path names of each audio file and its corresponding target class {'major': 0, 'minor': 1}. This dataframe will be used in a for loop to preprocess the audio data into spectrograms. 

**Audio File Details:**

Audio Files (Major): 502 

Audio Files (Minor): 357

Total Audio Files: 859

Percent of Major: 0.5844004656577415

Percent of Minor: 0.41559953434225844

Null-Baseline = 58.4%


**Data Description** 

| Attribute Name | Data Type | Description |
| -------------- | --------- | ----------- |
| `chord_qual`   | [object] | [the file path name of the audio file] |
| `target`   | [int64] | [a boolean of 0 or 1 determining the target class {'major': 0, 'minor': 1}] |



## EDA

We will create a dataframe composed of the file path names of each audio file and its corresponding target class {'major': 0, 'minor': 1}. This dataframe will be used in a for loop to preprocess the audio data into spectrograms. We will also explore the characteristics of raw audio files, spectrograms, and image files, to understand the fundamental differences between feeding spectrograms or images into a convolutional neural network. 

**Exploring Random Samples of Audio Data**
This (Bb Major 6th: Bb-D-F-G) chord will be difficult for our models to interpret. This is because all of the notes that go into a Major 6th chord are the same notes that go into it's relative Minor 7th chord (G Minor 7th: G-Bb-D-F). The ability to determine the difference between a Major 6th and it's relative Minor 7th chord is either a different instrument establishing the root or the context of the chords before and after it. Because we don't have these features to use in our models it seems that our models will be thrown off by this data. We can also notice inversions used in some of these audio files. This means that a G Minor 7th chord could validly have the notes ordered as (Bb-D-F-G) just as our Bb Major 6th chord is labeled.
For future consideration new data should be created or found with simpler chord structures and chord qualities that do not match even in inversions.

This F Diminished Triad (Note Order: F-Ab-B-F) does not match either chord quality classes. This is a diminished chord and distinctly different from the quality it was labeled as (minor). This is a mislabeled class and will impact the quality of our model.
Future consideration will be to locate other data that is properly labeled or create an original dataset with simpler chord qualities.


#### Differences

For images, the weights on the x and y axis are the same. Every point on an image represents a pixel intensity. For spectrograms, the x and y axis are fundamentally different. The x-axis represents time and the y_axis represents amplitude and frequency. This means that models will respond differently if a spectrogram is rotated. If the x-axis is now vertical, the model won't be able to adjust the axis observation.
Neighboring pixels in an image can strongly be assumed to be a part of the same object. However in spectrograms frequencies aren't separated by different objects but clumped together in a time-series. This means that a CNN model will have a harder time identifying separate sounds.


## Feature Extraction and Preprocessing 

We will be converting the audio wave files into augmented spectrograms and allocating them to their respective classes. We created the class, AudioTransformer, to load in an audio .wav file with a set sample rate of 22050 and setting the file to mono. Librosa offers to set these parameters as we load in the file which helps keep the formatting of the data consistent. The AudioTransformer class will be run through a for loop that reads each 859 unique audio files from the dataset created in the eda_audio notebook. The same_length function ensures all signal lengths are equal which will allow each spectrogram's x-axis (time) to be the same. The functions time_shift, transpose, and aug_sepctrogram are different methods of feature augmentation that were used to test the model. Different arrangements of functions were used and excluded to create a variety of different spectrograms for the model to be trained on.

Different Spectrograms used:

- Mel-Spectrograms
- Spectrograms
- Mel-Spectrograms with ticks and a colorbar
- Normalized spectrograms using StandardScalar


Feature augmentation to create more data:

- time_shift: shifts the audio to a random starting point within the set signal length of 50,000
- transpose: takes the audio signal and doubles it to be exactly on higher than its original
- aug_spectrogram: masks a bar of the y and x axis and random widths by setting those axis areas to 0. This is intended to prevent overfitting but did not perform well in my models.
  By creating a set of original spectrograms, a set with only time_shift, a set with only transpose, and a set with both transpose and time_shift, enabled me to quadruple my data.

Total Unique Spectrograms: 3,444

## Models 

**Summary**

We will develop three different kinds of image classification models to determine spectrogram images of minor and major chords.
We will start by creating a baseline convolutional neural network with two hidden layers. Our null baseline accuracy will be 58% because that is the higher proportion of our class data. We will determine a model's success by whether or not it exceeds our null baseline and baseline model. We will use regularization techniques to improve our baseline model, then use a VGG16 as a transfer learning method.

**We used two kinds of mel-spectrograms**

- Original Mel-Spectrograms: tested different formatting with the original spectrograms and found that removing tick marks, labels, and colorbars yielded the best results in our models.
- Scaled-Normalized Mel-Spectrograms: Scaled the signal time-series array of each spectrogram using the z-score method of StandardScaler

Metric = Accuracy 

Loss = Binary-Cross Entropy

#### CNN Models:

**base-model**: In our first CNN model we will only include two hidden layers with no regularization.

**scaled_base_model**: We will construct this model exactly like our first one but change the input data with scaled-normalized mel-spectrograms. This is to determine which kind of spectrogram formatting works the best with CNN models.

**added_layers_model**: We will add one more Conv2D and two more Dense hidden layers to allow the model to find more complex relationships in the data. We will also add an EarlyStopping with a large patience and only two dropout layers to lightly compensate the overfit base model. 


#### Transfer Learning Model: 

**VGG16_model**: The VGG16 is a transfer learning method model that has been pre trained on ImageNet. VGG16 specifically refers to a VGG model with 16 weight layers, including 13 convolutional layers and 3 fully connected layers. In this model, our spectrograms have to be processed specifically through the ImageDataGenerator. To do this we need to create a new file path that directs to two folders titled train and test. These folders consist of a major and minor folder with the same distribution as the original data source (Major: 58%, Minor: 42%). The test and train folders contain the same 75%/25% split as the other models. 


## Performance 

| Model Name | Accuracy | Loss |
| ---------- | -------- | ---- |
| `base-model` | 60.89% | 2.1 |
| `scaled_base_model` | 58.77% | 0.75 |
| `added_layers_model` | 60.16% | 0.67 |
| `VGG16_model` | 58.44% | 0.68 |


## Conclusion 

We tested scaled and unscaled spectrograms by running the same CNN structured model through both. Scaling our data using the z-score method stagnated the accuracy and loss results. For future consideration this project will aim to compare the performance of scaled and unscaled spectrogram data by running multiple iterations of CNN models through both to be able to compare a large series of results. Discovering the optimal way to format spectrogram data for a CNN model is still uncertain and this project aims to find a clear answer.

Although our added_layers_model performed the best and improved from our null_baseline, none of our models show significant clear results. There could be a number of reasons for this but two potential problems this project aims to tackle are the mislabeled class data discovered and the vague classes. After multiple random samplings of the audio files to confirm its class, I discovered some misclassified chords and unwanted chord qualities. Mislabeled data is the most obvious reason why our model could be performing poorly. For future consideration this project aims to generate our own audio data with completely correct class labels. We also aim to simplify the audio context by only recording triad chords with their respective inversions. Triad chords are much more distinguishable from one another than 7th chords. The majority of the samples listened to of the data used in this project were 7th chords causing potential overlap of chord qualities.


**Recommendations:**

- For audio classification models it is suggested to initially work with a smaller dataset. With a smaller dataset you can ensure each file is correctly labeled and have a better understanding of the different variations within each class. When audio files are turned into spectrograms they can be augmented with shifting and transposing to continually duplicate your input data. This helps compensate for the initial smaller dataset.

- Before scaling or normalizing your mel-spectrograms, run your CNN model using the original graphs. Our model comparison has shown that scaled spectrograms stagnate any increase in accuracy or decrease in overall loss. Future series of model comparisons will help solidify this recommendation. 






