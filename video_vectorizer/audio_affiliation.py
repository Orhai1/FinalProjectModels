import pandas as pd

def load_audio_id_set(csv_path = 'audio_ids.csv'):
    df = pd.read_csv(csv_path)
    return set(df['audio_id'].astype(str))

def check_tagged_audio(audio_id):
    affiliated_song_tags = load_audio_id_set()
    if audio_id in affiliated_song_tags:
        return {"uses_affiliated_song": 1}
    else:
        return {"uses_affiliated_song": 0}