from deepface import DeepFace

def create_embedding_db():
    # Build the leader DB embeddings
    DeepFace.find(
        img_path=None,
        db_path="leaders_dataset",        # Folder of leader images
        model_name="Facenet",             # can also try "ArcFace" or "VGG-Face"
        enforce_detection=False           # Disable if some images are tricky
    )

def find_leader(img_path):
    result = DeepFace.find(
        img_path=img_path,  # a new face to recognize
        db_path="leaders_dataset",
        model_name="Facenet",
        enforce_detection=True
    )

    if len(result) > 0 and not result[0].empty:
        top_match = result[0].iloc[0]
        print("Match found:", top_match["identity"], "(distance:", top_match["Facenet_cosine"], ")")
    else:
        print("No match found.")

if __name__ == "__main__":
    # create_embedding_db()
    find_leader("test_dataset/guy.webp")