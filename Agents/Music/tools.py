import re
from Agents.Music.spotify_client import SpotifyClient


class MusicTools:
    def __init__(self, spotify: SpotifyClient):
        self.spotify = spotify

    # ---------- level 1 ----------
    def play_track_by_query(self, query: str):
        query = query.strip()
        matches = self.spotify.search_track(query, limit=5)
        if not matches:
            return {"ok": False, "error": "no_track"}

        track = matches[0]
        self.spotify.play_track_uri(track["uri"])
        return {
            "ok": True,
            "track": {
                "name": track["name"],
                "artists": ", ".join(a["name"] for a in track["artists"]),
                "uri": track["uri"],
            },
        }

    # ---------- level 2 (cheat code) ----------
    def play_artist_this_is(self, artist_query: str):
        """
        Plays Spotify editorial playlist 'This Is {Artist}' if it exists.
        Fallback: plays artist context directly.
        """
        artist_query = artist_query.strip()
        if not artist_query:
            return {"ok": False, "error": "empty_artist"}

        # 1) resolve artist (canonical name + uri)
        artists = self.spotify.search_artist(artist_query, limit=5)
        if not artists:
            return {"ok": False, "error": "no_artist", "query": artist_query}

        artist = artists[0] or {}
        artist_name = artist.get("name", artist_query)
        artist_uri = artist.get("uri")

        # 2) search playlists for "This Is {Artist}"
        target = f"this is {artist_name}".lower()
        playlists = self.spotify.search_playlists(f'This Is {artist_name}', limit=10) or []

        best = None

        for p in playlists:
            if not isinstance(p, dict):
                continue
            pname = (p.get("name") or "").strip().lower()
            if pname == target:
                best = p
                break

        # near-match fallback
        if best is None:
            for p in playlists:
                if not isinstance(p, dict):
                    continue
                pname = (p.get("name") or "").lower()
                if "this is" in pname and artist_name.lower() in pname:
                    best = p
                    break

        # 3) play playlist context if found, else artist context
        if best is not None and best.get("uri"):
            self.spotify.start_playback_context(best["uri"], shuffle=True)
            return {
                "ok": True,
                "mode": "this_is_playlist",
                "artist": artist_name,
                "context": {"name": best.get("name"), "uri": best.get("uri")},
            }

        if artist_uri:
            self.spotify.start_playback_context(artist_uri, shuffle=True)
            return {
                "ok": True,
                "mode": "artist_context",
                "artist": artist_name,
                "context": {"uri": artist_uri},
            }

        return {"ok": False, "error": "no_context_uri", "artist": artist_name}

    # ---------- heuristic for fast path ----------
    def looks_like_direct_song_request(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False

        # "Song by Artist" is almost always a direct track request
        if " by " in t.lower():
            return True

        # quoted titles
        if t.startswith(("'", '"')) and t.endswith(("'", '"')):
            return True

        # if some is in the query, like "play some post malone"
        if " some" in t.lower():
            return False  # this is search for artist, not track

        # looks like a short title, not a sentence
        return bool(re.match(r'^[\w\s\'"\-\(\)\[\]\.,:&!/]+$', t)) and len(t.split()) >= 2
