import ollama
from Agents.Music.spotify_client import SpotifyClient
from Agents.Music.tools import MusicTools

class MusicAgent:
    def __init__(self):
        self.persona = (
            "You are Voxel-DJ, a charismatic, expressive, vibe-reading AI DJ. "
            "Speak with smooth, warm personality like Spotify’s DJ-X. "
            "2–4 sentences.\n"
        )
        self.spotify = SpotifyClient()
        self.tools = MusicTools(self.spotify)

    def handle(self, user_text: str):
        # Direct track
        if self.tools.looks_like_direct_song_request(user_text):
            r = self.tools.play_track_by_query(user_text)
            if r.get("ok"):
                t = r["track"]
                print(f"DJ-V: Putting on **{t['name']}** by {t['artists']}.")
            else:
                print("DJ-V: I couldn’t find that track.")
            return

        # Not a direct track => treat as artist request and use "This Is *Artist*" cheat
        r = self.tools.play_artist_this_is(user_text)
        if r.get("ok"):
            if r.get("mode") == "this_is_playlist":
                print(f"DJ-V: Shuffling **{r['context']['name']}** — cleanest way to get the artist’s hits.")
            else:
                print(f"DJ-V: Shuffling **{r['artist']}** — let’s run the discography vibe.")
        else:
            print("DJ-V: I couldn’t find that artist. Try being more specific.")
