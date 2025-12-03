"""Custom training callbacks."""

import base64
import hashlib
import os
import time
import requests
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class QuantConnectUploadCallback(Callback):
    """Upload best model to QuantConnect Object Store when training ends."""

    def __init__(self, org_id: str = None, key_prefix: str = "models"):
        super().__init__()
        self.org_id = org_id or os.environ.get("QUANTCONNECT_ORG_ID", "c084c68b78829375da3e09306b8c4b2c")
        self.key_prefix = key_prefix
        self.user_id = os.environ["QUANTCONNECT_USER_ID"]
        self.api_token = os.environ["QUANTCONNECT_API_TOKEN"]

    def on_fit_end(self, trainer, pl_module):
        best_path = trainer.checkpoint_callback.best_model_path
        if not best_path:
            print("No checkpoint to upload.")
            return

        filename = Path(best_path).name
        key = f"{self.key_prefix}/{filename}"

        print(f"\nUploading {filename} to QuantConnect Object Store...")
        result = self._upload(best_path, key)
        if result.get("success"):
            print(f"✓ Uploaded to: {key}")
        else:
            print(f"✗ Upload failed: {result}")

    def _get_auth_headers(self):
        """Generate timestamped authentication headers for QC API."""
        timestamp = str(int(time.time()))
        time_stamped_token = f"{self.api_token}:{timestamp}".encode("utf-8")
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        auth_bytes = f"{self.user_id}:{hashed_token}".encode("utf-8")
        auth_str = base64.b64encode(auth_bytes).decode("ascii")
        return {"Authorization": f"Basic {auth_str}", "Timestamp": timestamp}

    def _upload(self, filepath: str, key: str):
        with open(filepath, "rb") as f:
            file_data = f.read()

        headers = self._get_auth_headers()
        response = requests.post(
            "https://www.quantconnect.com/api/v2/object/set",
            headers=headers,
            data={"organizationId": self.org_id, "key": key},
            files={"objectData": file_data},
        )
        response.raise_for_status()
        return response.json()

