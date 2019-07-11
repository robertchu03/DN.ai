# DN.ai
Music Genre Classifier and Playlist Recommender System

## Summary

This project demonstrates the use of deep learning to classify music by genres and a distributed recommender system to suggest a playlist according to genre preferences of individual users. In a peer-to-peer network, each user can either be a requester or a service provider. Using a local self-learning chatbot, a user can make a request for music recommendation by simply stating a genre or a song title s/he likes. The system will then determine a peer machine that has the music collection most similar to the musical taste of the requester, prepare a recommended playlist and stream the music.

## Music Genre Classifier

First of all, we trained the genre classifier model with the [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html). This dataset consists of 1000 audio tracks with 100 songs from each of the 10 genres -- blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock. Using an open source library, [LibROSA](https://librosa.github.io/librosa/), we extracted the following 26 features from each track with a known genre and passed the features and labels of all 1000 songs to a 5-layer Artificial Neural Network (*Fig 1*), created using [Keras](https://keras.io/) and [scikit-learn](https://scikit-learn.org/stable/index.html) libraries, for training.

![GitHub Logo](/Images/genre%20classifier%20structure.png)
*Fig 1: Genre classifier ANN model structure*

##### 26 Features of Audio Track:
* 20 Mel-frequency cepstral coefficients (MFCC)
* Spectral centroid
* Spectral bandwidth
* Zero crossing rate
* Chroma frequencies
* Root mean square error
* Spectral roll-off frequency

The trained model was then distributed to all participating computers for 'predicting' genres of their own music collections.

We built a chatbot from a Python library, [ChatterBot](https://chatterbot.readthedocs.io/en/stable/index.html), and the chatbot web interface and the music streaming server from [Flask](http://flask.pocoo.org/docs/1.0/). The chatbot was trained with a corpus of customised dialogue data to facilitate its interactions with a user. We also implemented other logics to improve its understanding of certain keywords so that it can start communicating with music streaming servers whenever a user requests for music recommendations.

Upon starting up the streaming server for the first time, the program created a 26-feature array representing all the tracks from the local music collection. It then used the classifier to determine genres (*Fig 2*).

![GitHub Logo](/Images/classifying%20music%20library.png)
*Fig 2: Classifying local music libraries*

Next, a mid-point feature vector was generated for each genre by averaging all 26-feature vectors of the same kind. As a result, this set of mid-point vectors uniquely identifies the musical taste of the requestor.

## Playlist Recommender System

Music recommendation is done in a decentralised manner with the requesting and streaming servers contributing to some parts of the recommendation flow. When a user requests for a playlist by stating a particular genre (*Fig 3*), the genre and the corresponding mid-point feature vector are passed to all receiving servers (*Fig 4*).

![GitHub Logo](/Images/genre%20or%20reference%20track.png)
*Fig 3: User specifying genre or song title*

![GitHub Logo](/Images/passing%20genre%20and%20vector.png)
*Fig 4: Passing genre and feature vector to all servers*

Each server will then compute the Euclidean distances between the mid-point and the feature vectors of the same genre from its local repository and returns the mean distance to the requesting machine (*Fig 5*).

![GitHub Logo](/Images/euclidean%20distance.png)
*Fig 5: Returning mean Euclidean distances to client*

Finally the requesting machine selects the server with the shortest mean distance, which indicates the server with the music collection matching the requestor's taste, and plays the music from it (*Fig 6*).

![GitHub Logo](/Images/shortest%20distance.png)
*Fig 6: Selecting most closely matching server for music streaming*

The flow is similar in the case of the user providing a song title from his/her collection. The only difference is the actual feature vector of the song will be passed to the servers instead of the genre mid-point.
