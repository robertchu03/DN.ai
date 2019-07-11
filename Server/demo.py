import flask, flask.views
import os
import functools
from flask import jsonify, request

from pathlib import Path
import librosa
import numpy as np
import pickle
import shutil
import socket
import re
import sys
from statistics import mean

from scipy.spatial.distance import euclidean
from keras.models import model_from_json

###################
# FLASK WEBAPP CODE 
###################
app = flask.Flask(__name__)

app.secret_key = 'SECRET KEY'  # PROVIDE YOUR SECRET KEY

# login credential database
users = {'SOME USER NAME #1': 'SOME PASSWORD #1', 'SOME USER NAME #2': 'SOME PASSWORD #2'}  # PROVIDE LOGIN CREDENTIALS, ADD MORE IF NECESSARY

# Login page definition
class Main(flask.views.MethodView):
    def get(self):
        return flask.render_template('index.html')
    
    def post(self):
        if 'logout' in flask.request.form:
            flask.session.pop('username', None)
            return flask.redirect(flask.url_for('index'))
        required = ['username', 'passwd']
        for r in required:
            if r not in flask.request.form:
                flask.flash("Error: {} is required.".format(r))
                return flask.redirect(flask.url_for('index'))

        username = flask.request.form['username']
        passwd = flask.request.form['passwd']
        if username in users and users[username] == passwd:
            flask.session['username'] = username
        else:
            flask.flash("Username doesn't exist or incorrect password")
        return flask.redirect(flask.url_for('index'))

def login_required(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        if 'username' in flask.session:
            return method(*args, **kwargs)
        else:
            flask.flash("A login is required to see the page!")
            return flask.redirect(flask.url_for('index'))
    return wrapper

# Music page definition
class Music(flask.views.MethodView):
    @login_required
    def get(self):
        abs_songs = [] # to save absolute paths of recommended songs in original locations
        lib_songs = []  # to save relative paths of recommended songs in music library
        root = Path('.')  # root directory of web server

        # Load the dictionary of previously sorted distances
        dist_sorted = pickle.load(open("dist_sorted.pkl", "rb"))
        print(dist_sorted)
        
        # Populate abs_songs with original absolute paths
        for k in dist_sorted.keys():
            abs_songs.append(song_path[k])

        # Populate lib_songs with relative paths of songs from the music library
        for s in abs_songs:
            assert (sys.platform == 'darwin' or sys.platform == 'win32'), "Unsuitable OS used!!"
            # MacOS
            if sys.platform == 'darwin':
                song = str(s).split('/')[-1]  # extract file name from full path (original location)
                track = list(root.glob("**/" + song))[0]
                m = re.search(r'^.*\/(.*\/.*)$', str(track))
                lib_songs.append(m.group(1))            
            # Windows
            else:
                song = str(s).split('\\')[-1]  # extract file name from full path (original location)
                track = list(root.glob("**\\" + song))[0]
                m = re.search(r'^.*\\(.*\\.*)$', str(track))
                lib_songs.append(m.group(1).replace('\\', '/'))
        
        # Send recommended song list for rendering
        return flask.render_template('music.html', songs=lib_songs)

app.add_url_rule('/', view_func=Main.as_view('index'), methods=['GET', 'POST'])
app.add_url_rule('/music/', view_func=Music.as_view('music'), methods=['GET'])

########################
# NON-FLASK RELATED CODE 
########################
#
# ************************************************
# Step 1: Load Previously trained Genre Classifier
# ************************************************ 

# Load the json file that contains the model's structure
f = Path("music_genre_classifier_structure.json")
model_structure = f.read_text()

# Recreate music genre classifier model from json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("music_genre_classifier_weights.h5")

# *****************************************************
# Step 2: Create Feature Vectors of Unclassified Songs
# *****************************************************

# Create a list of absolute paths for all music tracks
assert (sys.platform == 'darwin' or sys.platform == 'win32'), "Unsuitable OS used!!"
# MacOS
if sys.platform == 'darwin':  # MacOS
    source = Path('[ROOT PATH TO YOUR OWN MUSIC LIBRARY]')  # <--- UPDATE THIS LINE B4 USING!!
    file_extension = "**/*.m4a"   # <--- UPDATE THIS LINE WITH THE RIGHT FILE EXTENSION B4 USING!!
else:  # Windows
    source = Path('[ROOT PATH TO YOUR OWN MUSIC LIBRARY]')  # <--- UPDATE THIS LINE B4 USING!!
    file_extension = "**\*.m4a"   # <--- UPDATE THIS LINE WITH THE RIGHT FILE EXTENSION B4 USING!!

song_path = [file for file in source.glob(file_extension)]

# Load the standardizer (scaler) from previously trained model
standardizer = pickle.load(open("scaler.pkl", "rb"))

def feature_extraction(song_path, scaler):
    """
    ACTION:
    -- Create an array of 26-feature vectors (standardized) from all songs listed in the song_path
    
    INPUT: 
    -- song_path = a list of absolute file paths of songs
    -- scaler = standardizer previously created from model training
    
    OUTPUT: 
    -- X_scaled = an array of 26-feature vectors (standardized)
    """
    array = []
    count = 0
    total = len(song_path)
    
    for song in song_path:
        count += 1
        print(f">>> Song #{count} of {total} <<<")
        print(song)
        
        # Extract 26 features (individual arrays) from each song
        y, sr = librosa.load(song)  # full length of song
        chroma_stft = librosa.feature.chroma_stft(y)  # chromagram
        rmse = librosa.feature.rmse(y)  # root-mean-square error (RMSE) value for each frame
        spec_cent = librosa.feature.spectral_centroid(y)
        spec_bw = librosa.feature.spectral_bandwidth(y)
        rolloff = librosa.feature.spectral_rolloff(y)
        zcr = librosa.feature.zero_crossing_rate(y)  
        mfcc = librosa.feature.mfcc(y)  # 20 Mel-frequency cepstral coefficients (MFCCs)

        # Create a 26-feature vector from the mean value of each feature array        
        vector = []
        vector.append(np.mean(chroma_stft))
        vector.append(np.mean(rmse))
        vector.append(np.mean(spec_cent))
        vector.append(np.mean(spec_bw))
        vector.append(np.mean(rolloff))
        vector.append(np.mean(zcr))

        for e in mfcc:
            vector.append(np.mean(e))
        
        array.append(vector)
    
    # Standardize 26-feature vectors with the trained scaler
    X = np.array(array)
    X_scaled = scaler.transform(X)
        
    return X_scaled

# "vec_library.npy" is an exported copy of an array that contains all 26-feature vectors 
# generated from the local music library. The file may exist. 
my_library = Path("vec_library.npy")

# If the feature array doesnt exist or isnt up-to-date, create this array (time-consuming)
if not (my_library.is_file() and len(np.load(my_library, allow_pickle=True))==len(song_path)):
    X = feature_extraction(song_path, standardizer)
    np.save(my_library, X)
else:
    X = np.load(my_library, allow_pickle=True)

# *********************************************
# Step 3: Predict Genres of Unclassified Songs
# ********************************************* 

# Load the genre label encoder from previously trained model
encoder = pickle.load(open("encoder.pkl", "rb"))

def genre_prediction(X, classifier, labelcoder):
    """
    ACTION:
    -- Predict genre of each song based on its standardized 26-feature vector in the array (X)
    
    INPUTS:
    -- X = an array containing standardized 26-feature vectors of all the songs
    -- classifier = genre classifier model (to predict genre)
    -- labelcoder = genre label encoder (to convert genre code to genre name)
    
    OUTPUT:
    -- genres = an array containing genre names of all the songs
    """
    prediction = classifier.predict(X)

    # Predict genre of each song with the highest probability
    g = [np.argmax(prediction[i]) for i in range(prediction.shape[0])]
    
    # Convert genre code to genre name for each song
    genres = encoder.inverse_transform(g)
    
    return genres

# "genres.npy" is an exported copy of an array that contains genres of all the songs in the local music library.
# This file may exist.
my_genres = Path("genres.npy")

# If the genre array doesnt exist or isnt up-to-date, create this array
if not (my_genres.is_file() and len(np.load(my_genres, allow_pickle=True))==len(song_path)):
    genres = genre_prediction(X, model, encoder)
    np.save(my_genres, genres)
else:
    genres = np.load(my_genres, allow_pickle=True)

# *****************************************
# Step 4: Create Music Library in Web Site 
# *****************************************
parent = Path('.')
library = parent/"static/music"

def create_library(parent_folder, song_loc, genres):
    """
    ACTION:
    -- Create music library, if not exists, under the web server root directory

    INPUTS:
    -- parent_folder = the library folder
    -- song_loc = a list of songs' absolute paths
    -- genres = an array of genres identified by the <genre_prediction> function

    OUTPUT:
    -- <NULL>
    """
    i = 0
    files = [x for x in parent_folder.glob('**/*.m4a')]  # UPDATE FILE EXTENSION IF NECESSARY

    # Only copy files when the library is empty or not up-to-date
    if len(song_loc) != len(files):
        
        # Clear the entire library
        shutil.rmtree(parent_folder, ignore_errors=True)
        
        # Create library
        for genre in genres:
            p = parent_folder/genre

            # Create folder if not exists
            if not p.exists():
                p.mkdir(parents=True)

            # Copy songs to designated folders
            shutil.copy(song_loc[i], str(p))
            i += 1

create_library(library, song_path, genres)

# *******************************************************
# Step 5: Find Mid-point Feature Vector of a given Genre 
# ******************************************************* 

def genre_midpoint_vector(genres, X):
    """
    PURPOSE:
    -- Mid-point vector will be used at the client side when requesting remote web servers for song recommendations.
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


def get_ip():
    """
    ACTION:
    -- Find IP address of the host (receiver) running this function
    
    INPUT:
    -- <NULL>
    
    OUTPUT:
    -- IP = Host IP address in a string
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


@app.route('/distance', methods=['POST', 'GET'])
def min_dest_distance():
    """
    ACTION:
    -- Given a genre (g), compute the Euclidean distance between a given feature vector (vec) and each vector of 
       the same genre from the music library of the receiving web server
    -- Return a dictionary that contains Web Server's IP and Mean Euclidean Distance, where distance = -1
       implies this server doesnt have any track of the requested genre
       
    VARIABLES:
    -- g = genre name either passed from source chatbot or determined from source reference track by 
            <genre_prediction> function
    -- vec = mid-point feature vector of the genre returned from the <genre_midpoint_vector> function or extracted 
            feature vector of a reference track from the <feature_extraction> function.
    -- destX = Standardized feature vector array of all the tracks in destination web server
    -- destGenres = genre array of all the tracks in destination web server
    
    OUTPUT:
    -- Dictionary {Web Server's IP: Mean Euclidean Distance}
    """
    input_params = request.get_json()
    g = input_params['g']  # passed from chatbot
    vec = np.array(input_params["vec"])  # passed from chatbot
    destX = X  # generated by function "feature_extraction()"
    destGenres = genres  # generated by function "genre_prediction()"
    
    # Initialization
    dist = {}  # dict for saving Euclidean distances
    mean_dist = -1  # for indicating this web server has no song of the requested genre (default)
    host_ip = get_ip()  # IP address of this web server
    
    # Obtain indices of all songs with the same genre (g)
    indices = np.where(destGenres==g)  # a one-element tuple containing indices as an array
    
    # Select feature vectors of all songs with the same genre (g)
    filtered_destX = destX[indices]
    
    # Calculate Euclidean distance between given vector and each vector in the destination server and
    # Save the distance with its correpsonding index in the dict (dist)
    for i, v in zip(indices[0], filtered_destX):
        dist[i] = dist.get(i, euclidean(v, vec))
        
    # Sort dict by distance in ascending order
    dist_sorted = dict(sorted(dist.items(), key=lambda kv:(kv[1], kv[0])))

    # Save sorted dict as a pickle file to be used later for music recommendation
    pickle.dump(dist_sorted, open("dist_sorted.pkl", "wb"))
    
    # Find the mean Euclidean distance
    if len(dist_sorted) > 0:  # dist_sorted not empty
        mean_dist = mean(dist_sorted.values())

    # mean_dist = -1 if dist_sorted is empty
    return jsonify({'host_ip': host_ip, 'mean_dist': mean_dist})


app.run(host='0.0.0.0', port=5001)