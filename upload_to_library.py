#!/usr/bin/env python3
"""
Upload model loader and feature engineering to a QuantConnect Library project.

This script creates/updates files in a QuantConnect Library project so that
all your algorithms in the organization can import them.

Usage:
    python upload_to_library.py --project-id YOUR_LIBRARY_PROJECT_ID
    
    # Or set environment variable:
    export QUANTCONNECT_LIBRARY_PROJECT_ID=123456
    python upload_to_library.py

First time setup:
    1. Create a Library project in QuantConnect Cloud
    2. Get the project ID from the URL (e.g., https://www.quantconnect.com/terminal/12345678)
    3. Run this script with that project ID
"""

import argparse
import base64
import hashlib
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# QuantConnect API endpoints
BASE_URL = "https://www.quantconnect.com/api/v2"


class QuantConnectAPI:
    """Simple QuantConnect API client."""
    
    def __init__(self):
        self.user_id = os.environ["QUANTCONNECT_USER_ID"]
        self.api_token = os.environ["QUANTCONNECT_API_TOKEN"]
    
    def _get_auth_headers(self):
        """Generate timestamped authentication headers."""
        timestamp = str(int(time.time()))
        time_stamped_token = f"{self.api_token}:{timestamp}".encode("utf-8")
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        auth_bytes = f"{self.user_id}:{hashed_token}".encode("utf-8")
        auth_str = base64.b64encode(auth_bytes).decode("ascii")
        return {"Authorization": f"Basic {auth_str}", "Timestamp": timestamp}
    
    def _post(self, endpoint: str, data: dict) -> dict:
        """Make a POST request to the QC API."""
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            headers=self._get_auth_headers(),
            json=data,
        )
        response.raise_for_status()
        return response.json()
    
    def list_files(self, project_id: int) -> list:
        """List all files in a project."""
        result = self._post("/files/read", {"projectId": project_id})
        if result.get("success"):
            return result.get("files", [])
        return []
    
    def create_file(self, project_id: int, name: str, content: str) -> dict:
        """Create a new file in a project."""
        return self._post("/files/create", {
            "projectId": project_id,
            "name": name,
            "content": content,
        })
    
    def update_file(self, project_id: int, name: str, content: str) -> dict:
        """Update an existing file in a project."""
        return self._post("/files/update", {
            "projectId": project_id,
            "name": name,
            "content": content,
        })
    
    def create_or_update_file(self, project_id: int, name: str, content: str) -> dict:
        """Create a file if it doesn't exist, otherwise update it."""
        existing_files = self.list_files(project_id)
        file_names = [f["name"] for f in existing_files]
        
        if name in file_names:
            print(f"  Updating existing file: {name}")
            return self.update_file(project_id, name, content)
        else:
            print(f"  Creating new file: {name}")
            return self.create_file(project_id, name, content)


def upload_library_files(project_id: int):
    """Upload model loader and feature engineering files to the library project."""
    api = QuantConnectAPI()
    
    # Files to upload
    files_to_upload = [
        ("model_loader.py", Path("src/inference/model_loader.py")),
        ("feature_engineering.py", Path("src/features/engineering.py")),
    ]
    
    print(f"\n{'='*60}")
    print(f"Uploading files to QuantConnect Library (Project ID: {project_id})")
    print(f"{'='*60}\n")
    
    success_count = 0
    for target_name, source_path in files_to_upload:
        if not source_path.exists():
            print(f"  ✗ Source file not found: {source_path}")
            continue
        
        content = source_path.read_text()
        
        try:
            result = api.create_or_update_file(project_id, target_name, content)
            if result.get("success"):
                print(f"  ✓ {target_name} uploaded successfully")
                success_count += 1
            else:
                print(f"  ✗ {target_name} failed: {result.get('errors', result)}")
        except Exception as e:
            print(f"  ✗ {target_name} error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Upload complete: {success_count}/{len(files_to_upload)} files uploaded")
    print(f"{'='*60}")
    
    if success_count == len(files_to_upload):
        print(f"""
✓ Library updated successfully!

To use in your algorithms:
─────────────────────────────────────────────────────────
1. Add the library to your algorithm project:
   - Open your algorithm in QuantConnect Cloud
   - Click "Add Library" in the project panel
   - Select your library project

2. Import in your main.py:
   from YourLibraryName.model_loader import TradingModelLoader
   from YourLibraryName.feature_engineering import engineer_features
─────────────────────────────────────────────────────────
""")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model loader to QuantConnect Library project"
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="QuantConnect Library project ID",
    )
    args = parser.parse_args()
    
    # Get project ID from args or environment
    project_id = args.project_id or os.environ.get("QUANTCONNECT_LIBRARY_PROJECT_ID")
    
    if not project_id:
        print("Error: No project ID provided.")
        print("\nPlease either:")
        print("  1. Pass --project-id YOUR_PROJECT_ID")
        print("  2. Set QUANTCONNECT_LIBRARY_PROJECT_ID in your .env file")
        print("\nTo get a project ID:")
        print("  1. Create a Library project in QuantConnect Cloud")
        print("  2. The project ID is in the URL: https://www.quantconnect.com/terminal/PROJECT_ID")
        return
    
    project_id = int(project_id)
    upload_library_files(project_id)


if __name__ == "__main__":
    main()

