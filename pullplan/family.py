"""A family of twelve service images built from packages installed on this host.

Each entry is the set of packages a service imports directly. The dependency
closure, and therefore the layer content, is resolved from installed metadata.
"""

FAMILY = {
    "llm-api-gateway": ["openai", "anthropic", "httpx", "fastapi", "uvicorn"],
    "embedding-service": ["transformers", "torch", "fastapi", "uvicorn"],
    "vector-search": ["chromadb", "fastapi", "uvicorn"],
    "onnx-inference": ["onnxruntime", "numpy", "fastapi"],
    "vision-preproc": ["opencv-python", "pillow", "numpy", "fastapi"],
    "audio-feature": ["librosa", "numpy", "scipy"],
    "tabular-train": ["scikit-learn", "pandas", "numpy", "scipy"],
    "tf-batch-scorer": ["tensorflow", "numpy"],
    "analytics-dash": ["streamlit", "plotly", "pandas"],
    "etl-arrow": ["pyarrow", "polars", "pandas"],
    "nlp-classic": ["nltk", "scikit-learn", "pandas"],
    "db-worker": ["sqlalchemy", "redis", "boto3"],
}
