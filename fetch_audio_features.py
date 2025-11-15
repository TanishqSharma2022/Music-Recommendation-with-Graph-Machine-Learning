import requests
import time
import pickle
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import json
from pathlib import Path as Data_Path

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'f448bf113f8748ada21eff77ba1dfda6')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', '141d20fe1b5145f1ac5d1dfc15506171')
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_AUDIO_FEATURES_URL = "https://api.spotify.com/v1/audio-features"

# Configuration
BATCH_SIZE = 100  # Spotify allows up to 100 tracks per request
TOKEN_EXPIRY_BUFFER = 300  # Refresh token 5 minutes before expiry (3600 - 300 = 3300s)
RATE_LIMIT_DELAY = 0.05  # 50ms between requests (conservative, Spotify allows more)
OUTPUT_FILE = "track_audio_features.pkl"
FAILED_TRACKS_FILE = "failed_tracks.pkl"

class SpotifyAudioFetcher:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry_time = None
        self.audio_features_cache = {}
        self.failed_tracks = set()
        
    def get_access_token(self) -> str:
        """Fetch a new access token from Spotify API."""
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = requests.post(SPOTIFY_TOKEN_URL, data=auth_data, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data['access_token']
            self.token_expiry_time = time.time() + token_data['expires_in']
            print(f"✓ New access token obtained. Expires in {token_data['expires_in']}s")
            return self.access_token
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"✗ Authentication failed (401): Invalid CLIENT_ID or CLIENT_SECRET")
                print(f"  Please regenerate credentials at: https://developer.spotify.com/dashboard")
            else:
                print(f"✗ Error fetching access token: {e}")
            raise
        except Exception as e:
            print(f"✗ Error fetching access token: {e}")
            raise
    
    def is_token_valid(self) -> bool:
        """Check if current token is still valid."""
        if self.access_token is None or self.token_expiry_time is None:
            return False
        return time.time() < (self.token_expiry_time - TOKEN_EXPIRY_BUFFER)
    
    def refresh_token_if_needed(self) -> None:
        """Refresh token if it's about to expire."""
        if not self.is_token_valid():
            print("⟳ Token expiring soon, refreshing...")
            self.get_access_token()
    
    def fetch_audio_features_batch(self, track_ids: List[str]) -> Dict[str, dict]:
        """
        Fetch audio features for tracks one at a time.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            Dictionary mapping track_id -> audio_features
        """
        self.refresh_token_if_needed()
        
        features_dict = {}
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        for track_id in track_ids:
            # Remove 'spotify:track:' prefix if present, keep only the ID
            clean_id = track_id.split(':')[-1]
            
            url = f"{SPOTIFY_AUDIO_FEATURES_URL}/{clean_id}"
            
            try:
                time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    retry_after = int(response.headers.get('Retry-After', 1))
                    print(f"⚠ Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    # Retry this track
                    response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 404:
                    # Track not found
                    self.failed_tracks.add(track_id)
                    continue
                
                response.raise_for_status()
                feature = response.json()
                
                if feature:
                    track_id_only = feature['id']
                    features_dict[track_id_only] = feature
                    # Also store with spotify:track: prefix for compatibility
                    features_dict[f"spotify:track:{track_id_only}"] = feature
            
            except requests.exceptions.RequestException as e:
                self.failed_tracks.add(track_id)
                continue
        
        return features_dict
    
    def fetch_all_features(self, track_ids: List[str], resume: bool = True) -> Dict[str, dict]:
        """
        Fetch audio features for all unique tracks one at a time.
        
        Args:
            track_ids: List of all track IDs (may contain duplicates)
            resume: If True, skip tracks already in cache
            
        Returns:
            Dictionary mapping track_id -> audio_features
        """
        # Get unique track IDs
        unique_ids = list(set(track_ids))
        
        # Load existing cache if resuming
        if resume and os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'rb') as f:
                    self.audio_features_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.audio_features_cache)} cached audio features")
            except Exception as e:
                print(f"✗ Error loading cache: {e}")
        
        # Load failed tracks from previous runs
        if os.path.exists(FAILED_TRACKS_FILE):
            try:
                with open(FAILED_TRACKS_FILE, 'rb') as f:
                    self.failed_tracks = pickle.load(f)
                print(f"⚠ Found {len(self.failed_tracks)} previously failed tracks")
            except Exception as e:
                print(f"✗ Error loading failed tracks: {e}")
        
        # Filter out already cached and failed tracks
        tracks_to_fetch = [
            tid for tid in unique_ids 
            if tid not in self.audio_features_cache and tid not in self.failed_tracks
        ]
        
        print(f"\nSummary:")
        print(f"  Total unique tracks: {len(unique_ids)}")
        print(f"  Already cached: {len(self.audio_features_cache)}")
        print(f"  Previously failed: {len(self.failed_tracks)}")
        print(f"  Need to fetch: {len(tracks_to_fetch)}\n")
        
        if len(tracks_to_fetch) == 0:
            print("✓ All tracks already have audio features!")
            return self.audio_features_cache
        
        # Get initial token
        if not self.is_token_valid():
            self.get_access_token()
        
        # Fetch one at a time
        for i in tqdm(range(0, len(tracks_to_fetch), BATCH_SIZE), desc="Fetching audio features"):
            batch = tracks_to_fetch[i:i + BATCH_SIZE]
            features = self.fetch_audio_features_batch(batch)
            self.audio_features_cache.update(features)
            
            # Save progress every 10 batches
            if (i // BATCH_SIZE) % 10 == 0:
                self.save_progress()
        
        # Final save
        self.save_progress()
        return self.audio_features_cache
    
    def save_progress(self) -> None:
        """Save current progress to disk."""
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(self.audio_features_cache, f)
        
        if self.failed_tracks:
            with open(FAILED_TRACKS_FILE, 'wb') as f:
                pickle.dump(self.failed_tracks, f)
        
        print(f"  ✓ Saved progress: {len(self.audio_features_cache)} features, "
              f"{len(self.failed_tracks)} failed")


def extract_track_ids_from_json(data_dir: str, n_files: int = None) -> List[str]:
    """
    Extract all unique track IDs from JSON files.
    
    Args:
        data_dir: Directory containing JSON playlist files
        n_files: Number of files to process (None = all)
        
    Returns:
        List of unique track IDs
    """
    data_path = Data_Path(data_dir)
    file_names = sorted(os.listdir(data_path))
    
    if n_files:
        file_names = file_names[:n_files]
    
    track_ids = set()
    
    for file_name in tqdm(file_names, desc="Extracting track IDs"):
        try:
            with open(data_path / file_name) as f:
                data = json.load(f)
            
            for playlist in data.get('playlists', []):
                for track in playlist.get('tracks', []):
                    track_uri = track.get('track_uri')
                    if track_uri:
                        # Extract only the ID part after the last colon
                        track_id = track_uri.split(':')[-1]
                        track_ids.add(track_id)
        except Exception as e:
            print(f"✗ Error processing {file_name}: {e}")
    
    return list(track_ids)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Spotify Audio Features Fetcher")
    print("=" * 60)
    
    # Step 1: Extract track IDs
    print("\n[1/2] Extracting track IDs from dataset...")
    DATA_DIR = 'spotify_million_playlist_dataset/data'
    N_FILES = 50  # Match your core.ipynb setting
    
    track_ids = extract_track_ids_from_json(DATA_DIR, N_FILES)
    print(f"✓ Found {len(track_ids)} unique tracks\n")
    
    # Step 2: Fetch audio features
    print("[2/2] Fetching audio features from Spotify API...")
    fetcher = SpotifyAudioFetcher(CLIENT_ID, CLIENT_SECRET)
    
    try:
        audio_features = fetcher.fetch_all_features(track_ids, resume=True)
        print(f"\n✓ Successfully fetched features for {len(audio_features)} tracks")
        print(f"✗ Failed to fetch {len(fetcher.failed_tracks)} tracks")
        print(f"\nResults saved to: {OUTPUT_FILE}")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user. Progress saved.")
        fetcher.save_progress()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        fetcher.save_progress()
        raise


if __name__ == "__main__":
    main()
