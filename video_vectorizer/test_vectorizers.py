from symbol_detection import detect_symbols_in_frames
from leader_detection import detect_leaders_in_frames
from text_affiliation import check_text_sources
from audio_affiliation import check_tagged_audio

# --- Inputs ---
video_id = "7299766556240743682"
frames_dir = f"../model_videos_frames/{video_id}_frames"

# Sample metadata for the video
description = "#طولكرم_فلسطين_سلفيت_نابلس_جنين #اريحا_الخليل_بيت_لحم_يافا_حيفا_عكا #ياسر_عرفات_ابو_عمار #fyp "
username = "abood._417"
audio_id = "7250927040110267138"

print("Symbol Detection Result:")
symbol_features = detect_symbols_in_frames(frames_dir)
for k, v in symbol_features.items():
    print(f"{k}: {v}")
print()

print("Leader Detection Result:")
leader_features = detect_leaders_in_frames(frames_dir)
for k, v in leader_features.items():
    print(f"{k}: {v}")
print()

print("Text Affiliation Result:")
text_features = check_text_sources(frames_dir, description, username)
for k, v in text_features.items():
    print(f"{k}: {v}")
print()

print("Audio Affiliation Result:")
audio_features = check_tagged_audio(audio_id)
for k, v in audio_features.items():
    print(f"{k}: {v}")
print()
