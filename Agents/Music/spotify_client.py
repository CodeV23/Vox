import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()


class SpotifyClient:
    def __init__(self):
        scope = " ".join([
            "user-read-recently-played",
            "user-top-read",
            "user-read-playback-state",
            "user-modify-playback-state",
        ])

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
                scope=scope,
                cache_path=os.getenv("SPOTIFY_CACHE_PATH", ".spotify_cache"),
                open_browser=True,
            )
        )

    # ---------- devices ----------
    def get_active_device_id(self):
        devices = self.sp.devices().get("devices", [])
        for d in devices:
            if d.get("is_active"):
                return d.get("id")
        return devices[0].get("id") if devices else None

    def set_shuffle(self, state: bool = True, device_id: str | None = None):
        if device_id is None:
            device_id = self.get_active_device_id()
        self.sp.shuffle(state, device_id=device_id)

    # ---------- search ----------
    def search_track(self, query: str, limit: int = 5):
        data = self.sp.search(q=query, type="track", limit=limit)
        return data.get("tracks", {}).get("items", [])

    def search_artist(self, query: str, limit: int = 5):
        data = self.sp.search(q=query, type="artist", limit=limit)
        return data.get("artists", {}).get("items", [])

    def search_playlists(self, query: str, limit: int = 10):
        data = self.sp.search(q=query, type="playlist", limit=limit)

        items = data.get("playlists", {}).get("items", []) or []
        # Sometimes Spotify/Spotipy can include None entries; filter them out.
        return [p for p in items if isinstance(p, dict)]

    # ---------- playback ----------
    def start_playback_context(self, context_uri: str, shuffle: bool = True):
        device_id = self.get_active_device_id()
        if not device_id:
            raise RuntimeError("No active Spotify device found. Open Spotify on a device first.")
        if shuffle:
            try:
                self.set_shuffle(True, device_id=device_id)
            except Exception:
                pass
        self.sp.start_playback(device_id=device_id, context_uri=context_uri)

    def play_track_uri(self, track_uri: str):
        device_id = self.get_active_device_id()
        if not device_id:
            raise RuntimeError("No active Spotify device found. Open Spotify on a device first.")
        self.sp.start_playback(device_id=device_id, uris=[track_uri])
