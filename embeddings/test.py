from pymilvus import connections, utility, Collection

# Connect to Milvus
connections.connect(host="192.168.20.156", port="19530")

# List all collections
all_collections = utility.list_collections()
print("📦 Existing collections:", all_collections)

# Loop and delete based on condition
for name in all_collections:
    try:
        col = Collection(name)
        num = col.num_entities
        print(f"🧮 Collection '{name}': {num} entities")

        # Example condition: delete if empty
        # if name in ['AIC24_batch1', 'AIC25_batch1', 'news_events', 'paper_collection2', 'test_rag_with_milvus', 'papers']:
        #     print(f"🗑️  Deleting collection '{name}'...")
        #     utility.drop_collection(name)

    except Exception as e:
        print(f"⚠️  Error with collection '{name}': {e}")
