def check_audio_tags(audio_tags):
    affiliated_song_tags = {"hamas nasheed", "fatah anthem"}
    return {"uses_affiliated_song": int(any(tag.lower() in affiliated_song_tags for tag in audio_tags))}
