from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import webbrowser
import os

from pathlib import Path
import numpy as np
import pickle
import requests
import socket
import sys

app = Flask(__name__)

mybot = ChatBot('Hawkeye')
trainer = ChatterBotCorpusTrainer(mybot)

# Assume MacOS or Windows are used
assert (sys.platform == 'darwin' or sys.platform == 'win32'), "Unsuitable OS used. Pls use either MacOS or Windows instead"

# Define corpus path for chatbot training
if sys.platform == 'darwin':  # MacOS
    corpus_path = './chatterbot_corpus/data/english'
else:  # Windows
    corpus_path = r'.\chatterbot_corpus\data\english'

for file in os.listdir(corpus_path):
    data = "{}/{}".format(corpus_path, file)
    trainer.train(data)

# Define local and remote machines
LOCAL_SERVER = ['localhost']
PEER_SERVERS = ['IP TO BE PROVIDED']  # <--- UPDATE THIS LINE B4 USING !!

# Load the genre label encoder from previously trained model
encoder = pickle.load(open("../Server/encoder.pkl", "rb"))

# Create a list of absolute paths for all music tracks
if sys.platform == 'darwin':  # MacOS
    source = Path('[ROOT PATH TO YOUR OWN MUSIC LIBRARY]')  # <--- UPDATE THIS LINE B4 USING!!
    file_extension = "**/*.m4a"   # <--- UPDATE THIS LINE WITH THE RIGHT FILE EXTENSION B4 USING!!
else:  # Windows
    source = Path('[ROOT PATH TO YOUR OWN MUSIC LIBRARY]')  # <--- UPDATE THIS LINE B4 USING!!
    file_extension = "**\*.m4a"   # <--- UPDATE THIS LINE WITH THE RIGHT FILE EXTENSION B4 USING!!

song_path = [file for file in source.glob(file_extension)]

# Reload feature vectors for all the songs in the client's local library 
my_library = Path("../Server/vec_library.npy")
X = np.load(my_library, allow_pickle=True)

# Reload classified genres for all the songs in the client's local library
my_genres = Path("../Server/genres.npy")
genres = np.load(my_genres, allow_pickle=True)

def get_audio_streaming_server(g, vec, target_servers):
    """
    ACTION:
    -- Select from the target_servers list an audio streaming server with the smallest mean Euclidean distance  
       from a given track (vec) of a particular genre (g).
    
    INPUTS:
    -- g = genre name either passed from source chatbot or determined from source reference track by 
           <genre_prediction> function
    -- vec = mid-point feature vector of the genre returned from the <genre_midpoint_vector> function or extracted 
              feature vector of a reference track from the <feature_extraction> function. 
    -- target_servers = a list of server IPs for connection
    
    OUTPUT:
    -- server = IP of server with the shortest mean Euclidean distance
    """
    
    results = []   # for keeping tuples (remote server ip, mean euclidean distance)
    data = {"g": g, "vec": vec.tolist()}
    
    # Connect to each remote server for min Euclidean distance
    for h in target_servers: 
        try:
            url = f"http://{h}:5001/distance"
            response = requests.post(url, json=data, timeout=5)  # timeout in 5 seconds
            result = response.json()
            results.append((result['host_ip'], result['mean_dist']))
        except:
            results.append((h, -1))  # distance = -1 represents a server cannot be reached
    
    # Sort servers by distance and return the one having the shortest distance
    results_sorted = sorted(results, key=lambda tup:tup[1])

    for tup in results_sorted:
        if tup[1] > -1:  # discard those with distance = -1
            return tup[0]

# Function to extract feature vector and classified genre from a reference track in client's local library
def reference_track_vector(song_title, song_path, X, genres):
    """
    ACTION:
    -- Extract feature vector and classified genre from a reference track in client's local library
    
    INPUTS:
    -- song_title = title of reference track
    -- song_path = list of songs' full paths in local music collection (original locations)
    -- X = feature vector array of all local songs
    -- genres = classified genres of all local songs
    
    OUTPUT:
    -- X[i] = feature vector of reference track, if found
    -- genre[i] = genre of reference track, if found
    """

    i = 0
    for s in song_path:  
        f = str(s).split('/')[-1].lower()  
        if song_title.lower() in f:
            return X[i], genres[i]
        else:
            i += 1
            if i == len(song_path):
                return None, None


def genre_midpoint_vector(genres, X):
    """
    PURPOSE:
    -- To be used at the client side when requesting remote web servers for song recommendations.
    -- When a reference track is not given, the client needs to find a mid-point feature vector from 
    -- all songs of that particular genre in the client's library. 

    ACTIONS:
    -- Compute and return the 26-feature mid-point vector of each genre found in the client's machine 
    
    INPUTS:
    -- genres = genre array returned by <genre_prediction> function
    -- X = standardized feature array returned by <feature_extraction> function
    
    Outputs: 
    -- mid_dict = dict{genres: mid-point vectors} for ease of reference
    """
    # Initialize a dictionary to record the 26-feature mid-point vector for each genre in source
    mid_dict = {}
       
    for g in set(genres):  # remove duplicate genres
        
        # Find indices that correspond to the given genre (g) in the genre array (genres)
        indices = np.where(genres==g)
    
        # Compute the mid-point vector for each genre
        mid = np.average(X[indices], axis=0)
        
        # Save the mid-point vector to dictionary (mid_dict)
        mid_dict[g] = mid
    
    return mid_dict

midpoint = genre_midpoint_vector(genres, X)


# Flask webapp code
@app.route("/")
def home():
    return render_template("index.html")

message = []

@app.route("/get")
def get_bot_response():

    # Save chatbot input to a list, allowing decisions to be made by chatbot based on diff stages of user input
    message.append((request.args.get('reply')).lower())
    userText = message[-1].lower()

    temp = ' '.join(message)
    split_message = temp.split()
    
    # Play music
    if any(w in split_message for w in ['song', 'music']):
        if 'collection' in split_message[:-1]:
            music = 'collection'
            machine = LOCAL_SERVER
        elif 'peers' in split_message[:-1]:
            music = 'peers'
            machine = PEER_SERVERS

        if 'genre' in split_message[:-1]:
            g = message[-1]
            if music == 'collection':
                v = midpoint.get(g, None)
            elif music == 'peers':
                # Client may wish to get song recommendation from peers for the genre not in his/her collection
                if g in encoder.classes_ and g not in set(genres):
                    v = midpoint.get(g, np.array([]))
                else:
                    v = midpoint.get(g, None)
            print("genre:", g)
            print("feature vector:", v)

            # Check for invalid genre
            if v is None:
                return "Invalid genre. Please provide again."

        elif 'track' in split_message[:-1]:
            title = message[-1]
            v, g = reference_track_vector(title, song_path, X, genres)

            # Check for invalid title
            if v is None:
                return "Invalid reference title. Please provide again."

        try:
            # Identify audio streaming server and generate recommended playlist
            server = get_audio_streaming_server(g, v, machine)

            # None refers to cases in whcih remote servers cannot be contacted or they dont have any suitable track
            if server is not None:
                url = f'http://{server}:5001'
                # Launch a new browser session to web server API
                webbrowser.open_new_tab(url)
                # Clean up the message list
                del message[:]            
                return "Have a nice day!!!"
            else:
                # Clean up the message list
                del message[:]
                return "No suitable tracks available from peers"

        except UnboundLocalError:
            pass  # Ignore this error as the value of g, v or machine may not exist

        return str(mybot.get_response(userText))
    
    else:
        # Clean up the message list
        del message[:]
        
        # Produce chatbot response to any request other than playing music
        return str(mybot.get_response(userText))


if __name__ == "__main__":
    app.run()
