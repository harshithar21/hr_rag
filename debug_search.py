"""Debug script: trace actual data flowing through the search pipeline."""
import sys
sys.path.insert(0, ".")
from backend.search import text_search, vector_search, embed_text, hybrid_search

# 1. Test embed
print("--- embed_text ---")
vec = embed_text("test query")
print(f"type={type(vec)}, len={len(vec)}, first_3={vec[:3]}")

# 2. Test text_search
print("\n--- text_search ---")
txt = text_search("test", limit=3)
print(f"type={type(txt)}, len={len(txt)}")
for row in txt:
    print(f"  row type={type(row)}, val={row}")

# 3. Test vector_search
print("\n--- vector_search ---")
vr = vector_search(vec, limit=3)
print(f"type={type(vr)}, len={len(vr)}")
for row in vr:
    print(f"  row type={type(row)}, val={row}")

# 4. Test hybrid_search
print("\n--- hybrid_search ---")
try:
    results = hybrid_search("test", top_k=3)
    print(f"OK, got {len(results)} results")
    for r in results:
        src = r.get("source")
        score = r.get("combined_score")
        print(f"  source={src}, score={score}")
except Exception as e:
    import traceback
    traceback.print_exc()
