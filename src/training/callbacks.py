"""Custom training callbacks."""

import base64
import hashlib
import os
import pickle
import time
import torch
from datetime import datetime
from pathlib import Path
import requests
from lightning.pytorch.callbacks import Callback


class QuantConnectUploadCallback(Callback):
    """Upload best model and all artifacts to QuantConnect Object Store when training ends."""

    def __init__(self, org_id: str = None, key_prefix: str = "models", train_dataset=None):
        super().__init__()
        self.org_id = org_id or os.environ.get("QUANTCONNECT_ORG_ID", "c084c68b78829375da3e09306b8c4b2c")
        self.key_prefix = key_prefix
        self.user_id = os.environ["QUANTCONNECT_USER_ID"]
        self.api_token = os.environ["QUANTCONNECT_API_TOKEN"]
        self.train_dataset = train_dataset  # Store reference to training dataset

    def on_fit_end(self, trainer, pl_module):
        best_path = trainer.checkpoint_callback.best_model_path
        if not best_path:
            print("No checkpoint to upload.")
            return

        print(f"\n{'='*50}")
        print("Preparing model package for QuantConnect...")
        print(f"{'='*50}")

        # Create unique folder name: patchtst-YYYYMMDD-HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"patchtst-{timestamp}"
        
        # Extract all artifacts to a temporary folder
        artifacts_dir = Path("checkpoints") / folder_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract model and all artifacts
            self._extract_artifacts(best_path, pl_module, artifacts_dir)
            
            # Upload all files in the folder
            print(f"\n{'='*50}")
            print(f"Uploading model package to QuantConnect: {folder_name}")
            print(f"{'='*50}")
            
            uploaded_files = []
            for file_path in artifacts_dir.glob("*"):
                if file_path.is_file():
                    relative_key = f"{self.key_prefix}/{folder_name}/{file_path.name}"
                    print(f"  Uploading {file_path.name}...")
                    result = self._upload(file_path, relative_key)
                    if result.get("success"):
                        uploaded_files.append(file_path.name)
                        print(f"    ✓ {file_path.name}")
                    else:
                        print(f"    ✗ {file_path.name}: {result}")
            
            if uploaded_files:
                print(f"\n✓ Successfully uploaded {len(uploaded_files)} files to: {self.key_prefix}/{folder_name}/")
                print(f"  Files: {', '.join(uploaded_files)}")
            else:
                print("\n✗ Upload failed for all files")
                
        except Exception as e:
            print(f"\n✗ Error preparing/uploading artifacts: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Optionally clean up local artifacts folder (comment out if you want to keep it)
            # import shutil
            # if artifacts_dir.exists():
            #     shutil.rmtree(artifacts_dir)
            pass

    def _extract_artifacts(self, checkpoint_path: Path, pl_module, output_dir: Path):
        """Extract model, scaler, and all necessary artifacts."""
        # Load the checkpoint to get the full model
        lightning_model = pl_module.__class__.load_from_checkpoint(checkpoint_path)
        pytorch_model = lightning_model.model
        pytorch_model.eval()
        
        hparams = lightning_model.hparams
        
        # 1. Save PyTorch model state dict
        model_path = output_dir / "model.pt"
        torch.save({
            'model_state_dict': pytorch_model.state_dict(),
            'hyperparameters': dict(hparams),
        }, model_path)
        print(f"  ✓ Saved model to {model_path.name}")
        
        # 2. Save scaler if available
        if self.train_dataset and hasattr(self.train_dataset, 'scaler'):
            scaler_path = output_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.train_dataset.scaler, f)
            print(f"  ✓ Saved scaler to {scaler_path.name}")
            
            # 3. Save feature columns
            if hasattr(self.train_dataset, 'feature_cols'):
                feature_cols_path = output_dir / "feature_columns.pkl"
                with open(feature_cols_path, 'wb') as f:
                    pickle.dump(self.train_dataset.feature_cols, f)
                print(f"  ✓ Saved feature columns to {feature_cols_path.name}")
                
                # 4. Save target normalization parameters
                if hasattr(self.train_dataset, 'target_mean') and hasattr(self.train_dataset, 'target_std'):
                    target_stats = {
                        'target_mean': self.train_dataset.target_mean,
                        'target_std': self.train_dataset.target_std,
                        'target_idx': self.train_dataset.target_idx,
                    }
                    target_stats_path = output_dir / "target_stats.pkl"
                    with open(target_stats_path, 'wb') as f:
                        pickle.dump(target_stats, f)
                    print(f"  ✓ Saved target stats to {target_stats_path.name}")
        else:
            print("  ⚠ Warning: Training dataset not available, scaler and features not saved")
        
        # 5. Save model architecture info (for reference)
        model_info = {
            'model_type': 'PatchTST',
            'hyperparameters': dict(hparams),
            'extraction_timestamp': datetime.now().isoformat(),
        }
        info_path = output_dir / "model_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"  ✓ Saved model info to {info_path.name}")

    def _get_auth_headers(self):
        """Generate timestamped authentication headers for QC API."""
        timestamp = str(int(time.time()))
        time_stamped_token = f"{self.api_token}:{timestamp}".encode("utf-8")
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        auth_bytes = f"{self.user_id}:{hashed_token}".encode("utf-8")
        auth_str = base64.b64encode(auth_bytes).decode("ascii")
        return {"Authorization": f"Basic {auth_str}", "Timestamp": timestamp}

    def _upload(self, filepath: Path, key: str):
        """Upload a file to QuantConnect Object Store."""
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

