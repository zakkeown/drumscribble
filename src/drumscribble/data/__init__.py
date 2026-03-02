"""Data loading pipelines for DrumscribbleCNN training."""
from drumscribble.data.parquet_loader import create_parquet_pipeline
from drumscribble.data.webdataset_loader import create_webdataset_pipeline

__all__ = ["create_parquet_pipeline", "create_webdataset_pipeline"]
