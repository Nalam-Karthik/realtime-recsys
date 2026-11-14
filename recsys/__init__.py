"""Two-stage real-time recommendation engine: two-tower retrieval + LambdaRank re-ranking."""
import os

# torch and faiss each ship an OpenMP runtime; two copies in one process segfault on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
