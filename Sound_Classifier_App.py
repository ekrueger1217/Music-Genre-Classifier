#!/usr/bin/env python
# coding: utf-8

# In[245]:


#import relevant packages
import streamlit as st
import numpy as np
import librosa
from librosa import display
from PIL import Image
import os
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
import io


# In[246]:


#load model. This model was built and trained in the 'Sound_Classifier_Models' notebook
model = load_model('model/sound_classifier.h5')

# In[247]:


#The model returns a numerical predicted class we'll use this mapping to return the corresponding label
class_mapping = {0: 'Speech', 1: 'Animal', 2: 'Vehicle', 3: 'Music'}


# In[248]:


#Create function that classifies a random ten second clip from a single audio file
def classify_sound(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=22050)
    
     # Randomly select a 10-second clip
    clip_duration = 10  # in seconds
    total_duration = librosa.get_duration(y=y, sr=sr)
    if total_duration <= clip_duration:
        start_time = 0.0
    else:
        start_time = random.uniform(0, total_duration - clip_duration)
        
    y_clip = y[int(start_time * sr):int((start_time + clip_duration) * sr)]
    
    # Compute the mel spectrogram
    M = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128, fmax=sr/2, n_fft=2048)
    M_db = librosa.power_to_db(M, ref=np.max)

    # Convert the mel spec to an image
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(M_db, cmap='inferno', x_axis='time', y_axis='mel')
    plt.axis('off')

    #capture image bytes and save in memory. This way user isn't saving a bunch of images on their machine.
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img_bytes.seek(0)

    #open image from bytes and preprocess (convert to rgb, resize to inpute size, normal per RGB, add batch dimension)
    img=Image.open(img_bytes)
    spec_resized = img.convert('RGB').resize((128, 128))
    spec_resized = np.array(spec_resized) / 255
    spec_reshaped = spec_resized.reshape(1, *spec_resized.shape)

    preds = model.predict(spec_reshaped)
    pred_class = np.argmax(preds)
    pred_label = class_mapping.get(pred_class, 'Unkown')

    return pred_label


# In[249]:
#loop through multiple audio files in a folder
def classify_files(files):
    class_counts = {'Speech': 0, 'Animal': 0, 'Vehicle': 0, 'Music': 0}
    results = []
    
    for file in files:
        pred_label = classify_sound(file)
        results.append((file.name, pred_label))
        class_counts[pred_label] += 1

    total_files = len(results)
    
    st.header("Results:")
    st.markdown("### Class Distribution:")
    class_distribution_data = {"Class": [], "Percentage": []}
    for class_label, count in class_counts.items():
        percentage = (count / total_files) * 100
        class_distribution_data["Class"].append(class_label)
        class_distribution_data["Percentage"].append(f"{percentage:.2f}%")

    st.table(class_distribution_data)

    st.markdown("### Classification Results:")
    classification_results_data = {"File": [], "Class": []}
    for audio_file, pred_label in results:
        classification_results_data["File"].append(audio_file)
        classification_results_data["Class"].append(pred_label)

    st.table(classification_results_data)
    
    return results


# In[250]:
def main():
    st.title("Sound Classification App")
    st.markdown("## What's making that sound?")

    uploaded_files = st.file_uploader("Upload Audio Files Here:", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)
    
    if uploaded_files:
        classify_files(uploaded_files)
 

if __name__ == "__main__":
    main()


# In[ ]:




