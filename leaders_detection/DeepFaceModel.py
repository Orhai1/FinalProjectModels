from deepface import DeepFace

DB_PATH = "leaders_dataset"

def create_embedding_db():
    """
    Create a database of face embeddings from the images in `db_path`.
    :return:
    """
    DeepFace.find(
        img_path=None,
        db_path=DB_PATH,
        model_name="Facenet",
        enforce_detection=False
    )

def find_leader(img_path, db_path=DB_PATH):
    """
    Check if an input face image matches any
    known faces in `db_path`. Prints match results.
    """
    result = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name="VGG-Face",
        enforce_detection=False
    )

    if len(result) > 0 and not result[0].empty:
        top_match = result[0].iloc[0]
        return top_match["identity"]
    else:
        return None

if __name__ == "__main__":
    # create_embedding_db()
    res = find_leader("test_dataset/guy.webp")

    print("Found match for: " + res) if res else print("No match found.")