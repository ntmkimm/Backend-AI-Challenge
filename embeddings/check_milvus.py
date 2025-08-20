from pymilvus import connections, utility

# Connect to Milvus server
connections.connect(host='192.168.20.156', port='19530')

# List all collections in Milvus
collections = utility.list_collections()
print("Collections in Milvus:", collections)

# for collection_name in collections:
#     try:
#         utility.drop_collection(collection_name)
#         print(f"Collection {collection_name} deleted successfully.")
#     except Exception as e:
#         print(f"Failed to delete collection {collection_name}: {e}")
# from pymilvus import Collection

# # Specify your collection name
# collection_name = "AIC25_fullbatch1"

# # Access the collection
# collection = Collection(collection_name)

# # Print collection schema
# print("Collection schema:", collection.schema)

# num_entities = collection.num_entities
# print(f"Number of entities in {collection_name}: {num_entities}")


# from pymilvus import connections, utility

# # Connect to Milvus
# connections.connect(host='192.168.20.156', port='19530')

# # List of collections to be deleted
# collections_to_delete = [
#     'testti_2691892',
#     'testti', 'testti_2770959', 'testti_3070168',
#     'testti_2652986', 'testti_normal', 'testti_optimized',
#      'test_ti', 'paper_collection', 'AIC25_fullbatch1_733118'
# ]

# # Drop collections one by one
# for collection_name in collections_to_delete:
#     try:
#         utility.drop_collection(collection_name)
#         print(f"Collection {collection_name} deleted successfully.")
#     except Exception as e:
#         print(f"Failed to delete collection {collection_name}: {e}")

