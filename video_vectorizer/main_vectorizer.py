from symbol_detection import detect_symbols_in_frames
from leader_detection import detect_leaders_in_frames
from text_affiliation import check_text_sources
from audio_affiliation import check_audio_tags

import numpy as np

def vectorize_video(frames_dir, description, username, audio_tags):
    features = {}

    features.update(detect_symbols_in_frames(frames_dir))
    features.update(detect_leaders_in_frames(frames_dir))
    features.update(check_text_sources(frames_dir, description, username))
    features.update(check_audio_tags(audio_tags))

    feature_order = list(features.keys())
    feature_vector = np.array([features[k] for k in feature_order])

    return feature_vector, feature_order
