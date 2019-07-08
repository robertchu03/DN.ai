# DN.ai
Music Genre Classifier and Audio Streaming Recommendation System

## Background
The objective is to allow users to request for music recommendation and streaming services from peer machines based on their genre preferences.  By using a distributed system of identical music genre classifier (ML model) in individual user computers in a peer-to-peer network, users can enjoy services among themselves without having their music preferences or other sensitive personal info leaked to tech giants for unsolicited commercial purposes.

## Implementation
The idea was to create a music genre classifier of a 5-layer artificial neural network, which was trained with a total of 1000 songs with 100 songs from each of the 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock).

Each peer computer would then receive a copy of the classifier for classifying all the songs from its local music library when starting up the audio streaming web server for the first time.  The classification process would only repeat upon startup when there are any changes in the music library.
