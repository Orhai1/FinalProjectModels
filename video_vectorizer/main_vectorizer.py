from symbol_detection import detect_symbols_in_frames
from leader_detection import detect_leaders_in_frames
from text_affiliation import check_text_sources
from audio_affiliation import check_tagged_audio

import numpy as np

def vectorize_video(frames_dir, description, username, audio_tags):
    features = {}
    metadata = {}

    symbol_feats, symbol_meta = detect_symbols_in_frames(frames_dir)
    leader_feats, leader_meta = detect_leaders_in_frames(frames_dir)
    text_feats, text_meta = check_text_sources(frames_dir, description, username)
    audio_feats = check_tagged_audio(audio_tags)

    features.update(symbol_feats)
    features.update(leader_feats)
    features.update(text_feats)
    features.update(audio_feats)

    metadata["symbols"] = symbol_meta
    metadata["leaders"] = leader_meta
    metadata["text"] = text_meta

    feature_order = list(features.keys())
    feature_vector = np.array([features[k] for k in feature_order])

    return feature_vector, feature_order, metadata
