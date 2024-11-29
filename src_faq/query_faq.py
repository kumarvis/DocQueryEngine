import numpy as np
from config.load_config import load_yaml_config
from src_faq.embedding_db_builder.save_load_embeddings import load_embeddings_index
from src_faq.embedding_db_builder.create_embedding import EmbeddingService

def map_user_query(config_path):
    config_dct = load_yaml_config(config_path)
    db_index = load_embeddings_index(config_dct)
    embedding_service_obj = EmbeddingService(config_path)
    dist_thresh = config_dct["dist_thresh"]
    while True:
        # Get the query from the user
        query = input("Enter your query (or type 'bye' to exit): ")
        if query.lower() == "bye":
                print("Goodbye!")
                break
        query_embedding = embedding_service_obj.get_embeddings([query])
        norm_query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        distances, indices = db_index.search(norm_query_embedding.reshape(1, -1), k=3)
        index = int(indices[0][0])
        dist = float(distances[0][0])
        if dist < dist_thresh :
            print(f"Map/Refer to Question No {index+1} with Distance = {dist}")
        else :
            print(f"I’m sorry, that’s out of my area of expertise. Please reach out to our team")
            print(f"Map/Refer to Question No {index+1} with Distance = {dist}")

if __name__ == "__main__":
    config_path = "config/config_faq.yaml"
    map_user_query(config_path)
    