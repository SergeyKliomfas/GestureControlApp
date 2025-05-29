import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os


load_dotenv()

scope = "user-modify-playback-state"

scope = "user-modify-playback-state"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope=scope
))


def play():
    sp.start_playback()

def pause():
    sp.pause_playback()

def next_track():
    sp.next_track()

def previous_track():
    sp.previous_track()

def volume_up():
    current_volume = sp.current_playback()['device']['volume_percent']
    sp.volume(min(current_volume + 10, 100))

def volume_down():
    current_volume = sp.current_playback()['device']['volume_percent']
    sp.volume(max(current_volume - 10, 0))

