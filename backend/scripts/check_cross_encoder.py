import sys
from sentence_transformers import CrossEncoder

model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
try:
    ce = CrossEncoder(model_name, local_files_only=True)
    print('Cross-encoder model found locally.')
except Exception as e:
    print('Cross-encoder NOT found locally:', e)
    sys.exit(1)
