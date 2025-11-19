import torch
import torch_geometric
# General libraries
import json
from pathlib import Path as Data_Path
import os
from os.path import isfile, join
import pickle
import random

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
# %matplotlib inline

from tqdm import tqdm
# Import relevant ML libraries
from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, ModuleList, Linear
import torch.nn.functional as F

import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv, GATConv, SAGEConv
from torch_geometric.typing import Adj, OptTensor, SparseTensor

print(f"Torch version: {torch.__version__}; Torch-cuda version: {torch.version.cuda}; Torch Geometric version: {torch_geometric.__version__}.")
# set the seed for reproducibility
seed = 224
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
MAIN_DIR = "/home/shaileshk/Desktop/Shailesh/Sem_7/ML_w_Graphs/MLG/"
DATA_DIR = Data_Path('spotify_million_playlist_dataset/data')
os.chdir(MAIN_DIR)
with open(f"{DATA_DIR}/{os.listdir(DATA_DIR)[0]}") as jf:
  example_file = json.load(jf)

print(example_file['playlists'][0])
"""
Here we define classes for the data that we are going to load. The data is stored in JSON files, each
which contain playlists, which themselves contain tracks. Thus, we define three classes:
  Track       --> contains information for a specific track (its id, name, etc.)
  Playlist    --> contains information for a specific playlist (its id, name, etc. as well as a list of Tracks)
  JSONFile    --> contains the loaded json file and stores a dictionary of all of the Playlists

Note: if we were to use the artist information, we could make an Artist class
"""

class Track:
  """
  Simple class for a track, containing its attributes:
    1. URI (a unique id)
    2. Name
    3. Artist info (URI and name)
    4. Parent playlist
  """

  def __init__(self, track_dict, playlist):
    self.uri = track_dict["track_uri"]
    self.name = track_dict["track_name"]
    self.artist_uri = track_dict["artist_uri"]
    self.artist_name = track_dict["artist_name"]
    self.album_uri = track_dict["album_uri"]
    self.album_name = track_dict["album_name"]
    self.playlist = playlist

  def __str__(self):
    return f"Track {self.uri} called {self.name} by {self.artist_uri} ({self.artist_name}) and in {self.album_uri} | {self.album_name} in playlist {self.playlist}."

  def __repr__(self):
    return f"Track {self.uri}"

class Playlist:
  """
  Simple class for a playlist, containing its attributes:
    1. Name (playlist and its associated index)
    2. Title (playlist title in the Spotify dataset)
    3. Loaded dictionary from the raw json for the playlist
    4. Dictionary of tracks (track_uri : Track), populated by .load_tracks()
    5. List of artists uris
  """

  def __init__(self, json_data, index):

    self.name = f"playlist_{index}"
    self.title = json_data["name"]
    self.data = json_data

    self.tracks = {}

  def load_tracks(self):
    """ Call this function to load all of the tracks in the json data for the playlist."""

    tracks_list = self.data["tracks"]
    self.tracks = {x["track_uri"] : Track(x, self.name) for x in tracks_list}


  def __str__(self):
    return f"Playlist {self.name} with {len(self.tracks)} tracks loaded."

  def __repr__(self):
    return f"Playlist {self.name}"

class JSONFile:
  """
  Simple class for a JSON file, containing its attributes:
    1. File Name
    2. Index to begin numbering playlists at
    3. Loaded dictionary from the raw json for the full file
    4. Dictionary of playlists (name : Playlist), populated by .process_file()
  """

  def __init__(self, data_path, file_name, start_index):

    self.file_name = file_name
    self.start_index = start_index

    with open(join(data_path, file_name)) as json_file:
      json_data = json.load(json_file)
    self.data = json_data

    self.playlists = {}

  def process_file(self):
    """ Call this function to load all of the playlists in the json data."""

    for i, playlist_json in enumerate(self.data["playlists"]):
      playlist = Playlist(playlist_json, self.start_index + i)
      playlist.load_tracks()
      self.playlists[playlist.name] = playlist

  def __str__(self):
    return f"JSON {self.file_name} has {len(self.playlists)} playlists loaded."

  def __repr__(self):
    return self.file_name

DATA_PATH = Data_Path('spotify_million_playlist_dataset/data')
N_FILES_TO_USE = 50

file_names = sorted(os.listdir(DATA_PATH))
file_names_to_use = file_names[:N_FILES_TO_USE]

n_playlists = 0

# load each json file, and store it in a list of files
JSONs = []
for file_name in tqdm(file_names_to_use, desc='Files processed: ', unit='files', total=len(file_names_to_use)):
  json_file = JSONFile(DATA_PATH, file_name, n_playlists)
  json_file.process_file()
  n_playlists += len(json_file.playlists)
  JSONs.append(json_file)
  playlist_data = {}
track_data = []
playlists = []
tracks = []

# build list of all unique playlists, tracks
for json_file in tqdm(JSONs):
  playlists += [p.name for p in json_file.playlists.values()]
  tracks += [track.uri for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
  playlist_data = playlist_data | json_file.playlists

  ## create graph from these lists

# adding nodes
G = nx.Graph()
G.add_nodes_from([
    (p, {'name':p, "node_type" : "playlist"}) for p in playlists
])
G.add_nodes_from([
    (t, {'name':t, "node_type" : "track"}) for t in tracks
])
# add node types of track to album and artist
G.add_nodes_from([
    (track.album_uri, {'name':track.album_uri, "node_type" : "album"}) for p_name, playlist in playlist_data.items() for track in playlist.tracks.values()
])
G.add_nodes_from([
    (track.artist_uri, {'name':track.artist_uri, "node_type" : "artist"}) for p_name, playlist in playlist_data.items() for track in playlist.tracks.values()   
])

# adding edges
track_edge_list = []
album_edge_list = []
artist_edge_list = []
for p_name, playlist in playlist_data.items():
  for track in playlist.tracks.values():
    track_edge_list.append((p_name, track.uri))
    album_edge_list.append((track.uri, track.album_uri))
    artist_edge_list.append((track.uri, track.artist_uri))

G.add_edges_from(track_edge_list, edge_types="track_in_playlist")
G.add_edges_from(album_edge_list, edge_types="track_in_album")
G.add_edges_from(artist_edge_list, edge_types="track_by_artist")

print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges())

from collections import Counter

cnt = Counter([d["node_type"] for (_, d) in G.nodes(data=True)])
print(cnt)

cmap_theme = "Set1"

import random
# -------- CONFIG --------
NUM_PLAYLISTS_TO_SAMPLE = 5000

# -------- STEP 1: sample playlists --------
playlist_nodes = [
    n for n, d in G.nodes(data=True) if d["node_type"] == "playlist"
]

sampled_playlists = random.sample(playlist_nodes, NUM_PLAYLISTS_TO_SAMPLE)

sub_nodes = set(sampled_playlists)

# -------- STEP 2: playlist → track --------
for p in sampled_playlists:
    for neigh in G.neighbors(p):
        if G.nodes[neigh]["node_type"] == "track":
            sub_nodes.add(neigh)

# -------- STEP 3: track → album & artist ONLY --------
for t in list(sub_nodes):
    if G.nodes[t]["node_type"] == "track":
        for neigh in G.neighbors(t):
            if G.nodes[neigh]["node_type"] in ["album", "artist"]:
                sub_nodes.add(neigh)

# -------- STEP 4: build subgraph --------
sub_G = G.subgraph(sub_nodes).copy()

print("Nodes:", sub_G.number_of_nodes())
print("Edges:", sub_G.number_of_edges())

from collections import Counter

cnt = Counter([d["node_type"] for (_, d) in sub_G.nodes(data=True)])
print(cnt)

with open("5K_playlist_graph.pkl", "wb") as f:
    pickle.dump(sub_G, f)