"""Blob store for large tracing content (prompts, responses, code, etc.)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Union


class BlobStore:
    """Content-addressable storage for large trace payloads."""

    def __init__(self, base_path: str = "traces/blobs"):
        self.base_path = Path(base_path)

    def put(self, content: Union[str, dict, list], ext: str = "txt") -> str:
        """
        Store content and return blob reference.

        Args:
            content: String content or JSON-serializable object
            ext: File extension ('txt' or 'json')

        Returns:
            Blob reference in format 'blob:sha256:<hash>'
        """
        # Canonicalize content
        if isinstance(content, (dict, list)):
            canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
            ext = "json"
        else:
            canonical = str(content)

        # Compute SHA256 hash
        content_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

        # Construct file path
        blob_dir = self.base_path / "sha256"
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blob_dir / f"{content_hash}.{ext}"

        # Write if doesn't exist (idempotent)
        if not blob_path.exists():
            try:
                with open(blob_path, 'w', encoding='utf-8') as f:
                    f.write(canonical)
            except Exception:
                # Safe: don't break run if blob storage fails
                pass

        return f"blob:sha256:{content_hash}"

    def get(self, ref: str) -> str:
        """
        Retrieve blob content by reference.

        Args:
            ref: Blob reference in format 'blob:sha256:<hash>'

        Returns:
            Content as string
        """
        if not ref.startswith("blob:sha256:"):
            raise ValueError(f"Invalid blob reference: {ref}")

        content_hash = ref.split(":")[-1]

        # Try both extensions
        for ext in ["txt", "json"]:
            blob_path = self.base_path / "sha256" / f"{content_hash}.{ext}"
            if blob_path.exists():
                with open(blob_path, 'r', encoding='utf-8') as f:
                    return f.read()

        raise FileNotFoundError(f"Blob not found: {ref}")
