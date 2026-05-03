import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("services.embedding_service")

model = None

def _get_model():
    global model
    if model is None:
        try:
            logger.info("Initializing SentenceTransformer embedding model (Lazy Load)...")
            # Set local_files_only=True to skip internet check if already downloaded
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    return model

# Simple In-memory cache for embeddings to avoid CPU-heavy re-calculations in loops
_embedding_cache = {}

def embed_text(text: str) -> list[float]:
    """
    Computes a 384-dimensional vector embedding for the input text.
    Uses an internal cache to avoid redundant compute for the same strings.
    """
    if not text or not isinstance(text, str):
        return []
    
    # normalize key
    key = text.strip().lower()
    if key in _embedding_cache:
        return _embedding_cache[key]

    transformer = _get_model()
    if not transformer:
        return []
        
    vec = transformer.encode(text).tolist()
    
    # Basic cache management (prevent memory leak)
    if len(_embedding_cache) > 1000:
        _embedding_cache.clear()
        
    _embedding_cache[key] = vec
    return vec

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Mathematical cosine similarity between two vectors."""
    import numpy as np
    vec_a = np.array(a)
    vec_b = np.array(b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def semantic_match(text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity score between two strings.
    Higher is more similar (0.0 to 1.0).
    """
    if not text1 or not text2:
        return 0.0
    
    # Perfect string match = 1.0 (Bypass embedding)
    if text1.strip().lower() == text2.strip().lower():
        return 1.0
        
    v1 = embed_text(text1)
    v2 = embed_text(text2)
    
    if not v1 or not v2:
        return 0.0
        
    return cosine_similarity(v1, v2)

def are_equivalent_occupations(user_occ: str, rule_occ: str) -> bool:
    """
    Checks if two occupation terms are equivalent in a welfare context.
    Rule: Strict match first, then embedding similarity fallback.
    """
    if not user_occ or not rule_occ:
        return False
        
    u = user_occ.strip().lower()
    r = rule_occ.strip().lower()
    
    # 1. Strict Match
    if u == r:
        return True
        
    # 2. High-precision embedding similarity (deterministic given model weights)
    # Threshold 0.85 is used to capture minor variations (e.g. 'farmer' vs 'farmers')
    score = semantic_match(u, r)
    if score > 0.85:
        logger.debug(f"Deterministic semantic match: '{u}' ~ '{r}' (Score: {score:.4f})")
        return True
        
    return False
