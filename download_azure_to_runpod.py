"""
Azure Blob Storage to RunPod Volume Downloader

This script downloads all data from a specified Azure Blob Storage container prefix
(rsna_drrs_and_nifti) to a local directory on RunPod storage volume, preserving
the complete folder structure.

Features:
- Preserves full directory structure
- Concurrent downloads for speed
- Progress tracking with tqdm
- Resume capability (skips already downloaded files)
- Robust error handling and logging

Usage:
    python download_azure_to_runpod.py
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from azure.storage.blob import BlobServiceClient


# --- Configuration -----------------------------------------------------------
@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for the download pipeline."""
    
    # Azure Blob Storage
    BLOB_CONNECTION_STRING: str = (
        "DefaultEndpointsProtocol=https;AccountName=spartis9488473038;"
        "AccountKey=WxiLwTEm+WEut0AIFRTLiWcXgHhDixXtYtF5gbbGIKLMWANt5wHOVwg/"
        "QzRgz2uG1CHcazDil58i+ASttN+yaA==;EndpointSuffix=core.windows.net"
    )
    CONTAINER_NAME: str = "ct-big-data"
    
    # Source prefix in Azure (what to download)
    SOURCE_PREFIX: str = "rsna_drrs_and_nifti"
    
    # Destination directory on RunPod storage volume
    # Modify this path to match your RunPod volume mount point
    DESTINATION_DIR: Path = Path("/workspace/datasets/rsna_drrs_and_nifti")
    
    # Concurrency settings
    MAX_CONCURRENT_DOWNLOADS: int = 10  # Adjust based on network bandwidth
    
    # Resume settings
    SKIP_EXISTING: bool = True  # Skip files that already exist locally
    VERIFY_SIZE: bool = True    # Verify file size matches before skipping


# --- Logging Setup -----------------------------------------------------------
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- Download Functions ------------------------------------------------------
def download_blob_with_structure(
    blob_name: str,
    source_prefix: str,
    destination_root: Path,
    container_client,
    verify_size: bool = True
) -> tuple[str, bool, str]:
    """
    Download a single blob and preserve its directory structure.
    
    Args:
        blob_name: Full blob name in Azure
        source_prefix: The prefix to strip from blob_name to get relative path
        destination_root: Root directory where files should be saved
        container_client: Azure container client
        verify_size: Whether to verify file size before skipping
        
    Returns:
        Tuple of (blob_name, success, message)
    """
    try:
        # Calculate relative path by removing the source prefix
        if blob_name.startswith(source_prefix):
            relative_path = blob_name[len(source_prefix):].lstrip('/')
        else:
            relative_path = blob_name
        
        # Construct local file path
        local_path = destination_root / relative_path
        
        # Check if file already exists
        if local_path.exists():
            if verify_size:
                # Get blob properties to check size
                blob_client = container_client.get_blob_client(blob_name)
                blob_properties = blob_client.get_blob_properties()
                remote_size = blob_properties.size
                local_size = local_path.stat().st_size
                
                if local_size == remote_size:
                    return blob_name, True, f"⏭️ Skipped (already exists with correct size)"
                else:
                    logger.warning(
                        f"Size mismatch for {local_path.name}: "
                        f"local={local_size}, remote={remote_size}. Re-downloading."
                    )
            else:
                return blob_name, True, f"⏭️ Skipped (already exists)"
        
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the blob
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)
        
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        return blob_name, True, f"✅ Downloaded ({file_size_mb:.2f} MB)"
        
    except Exception as e:
        error_msg = f"❌ Failed: {str(e)}"
        logger.error(f"Error downloading {blob_name}: {e}")
        return blob_name, False, error_msg


def get_blob_list(container_client, prefix: str) -> list[str]:
    """
    Get list of all blobs under the specified prefix.
    
    Args:
        container_client: Azure container client
        prefix: Blob prefix to filter by
        
    Returns:
        List of blob names
    """
    logger.info(f"📋 Listing all blobs under prefix: {prefix}")
    
    blobs = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        # Only include actual files (blobs with content)
        if not blob.name.endswith('/'):
            blobs.append(blob.name)
    
    logger.info(f"📊 Found {len(blobs)} files to download")
    return blobs


def calculate_total_size(container_client, blob_names: list[str]) -> float:
    """
    Calculate total size of all blobs in GB.
    
    Args:
        container_client: Azure container client
        blob_names: List of blob names
        
    Returns:
        Total size in GB
    """
    logger.info("📐 Calculating total download size...")
    total_bytes = 0
    
    for blob_name in tqdm(blob_names, desc="Calculating size", unit="file"):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            total_bytes += properties.size
        except Exception as e:
            logger.warning(f"Could not get size for {blob_name}: {e}")
    
    total_gb = total_bytes / (1024**3)
    logger.info(f"📦 Total download size: {total_gb:.2f} GB ({total_bytes:,} bytes)")
    return total_gb


# --- Main Pipeline -----------------------------------------------------------
def main():
    """Main download pipeline."""
    config = DownloadConfig()
    
    logger.info("=" * 60)
    logger.info("=== AZURE BLOB STORAGE TO RUNPOD DOWNLOADER ===")
    logger.info("=" * 60)
    logger.info(f"📥 Source: {config.CONTAINER_NAME}/{config.SOURCE_PREFIX}")
    logger.info(f"📁 Destination: {config.DESTINATION_DIR}")
    logger.info(f"🔧 Concurrent downloads: {config.MAX_CONCURRENT_DOWNLOADS}")
    logger.info(f"🔄 Skip existing: {config.SKIP_EXISTING}")
    logger.info("=" * 60)
    
    # Create destination directory
    config.DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Destination directory ready: {config.DESTINATION_DIR}")
    
    # Connect to Azure Blob Storage
    try:
        logger.info("🔗 Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(
            config.BLOB_CONNECTION_STRING
        )
        container_client = blob_service_client.get_container_client(
            config.CONTAINER_NAME
        )
        logger.info("✅ Connected to Azure Blob Storage")
    except Exception as e:
        logger.critical(f"❌ Failed to connect to Azure Storage: {e}")
        return
    
    # Get list of all blobs to download
    try:
        blob_list = get_blob_list(container_client, config.SOURCE_PREFIX)
    except Exception as e:
        logger.critical(f"❌ Failed to list blobs: {e}")
        return
    
    if not blob_list:
        logger.warning("⚠️ No files found to download. Exiting.")
        return
    
    # Skip size calculation - start downloading immediately
    logger.info("⚡ Skipping size calculation - starting download immediately!")
    
    # Download all blobs with concurrent workers
    logger.info(f"🚀 Starting download of {len(blob_list)} files...")
    
    results = []
    with ThreadPoolExecutor(
        max_workers=config.MAX_CONCURRENT_DOWNLOADS,
        thread_name_prefix='Downloader'
    ) as executor:
        # Submit all download tasks
        future_to_blob = {
            executor.submit(
                download_blob_with_structure,
                blob_name,
                config.SOURCE_PREFIX,
                config.DESTINATION_DIR,
                container_client,
                config.VERIFY_SIZE
            ): blob_name
            for blob_name in blob_list
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(blob_list), desc="Downloading", unit="file") as pbar:
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                try:
                    blob_name, success, message = future.result()
                    results.append({
                        "blob": blob_name,
                        "success": success,
                        "message": message
                    })
                    
                    # Update progress bar description with last file
                    pbar.set_postfix_str(Path(blob_name).name[:40])
                    pbar.update(1)
                    
                except Exception as exc:
                    error_msg = f"❌ Unhandled exception: {exc}"
                    results.append({
                        "blob": blob_name,
                        "success": False,
                        "message": error_msg
                    })
                    logger.error(f"Exception for {blob_name}: {exc}")
                    pbar.update(1)
    
    # Summary
    success_count = sum(1 for r in results if r["success"])
    skipped_count = sum(1 for r in results if "Skipped" in r["message"])
    failed_count = len(results) - success_count
    
    logger.info("\n" + "=" * 60)
    logger.info("=== DOWNLOAD SUMMARY ===")
    logger.info("=" * 60)
    logger.info(f"📊 Total files: {len(results)}")
    logger.info(f"✅ Successfully downloaded: {success_count - skipped_count}")
    logger.info(f"⏭️ Skipped (already exist): {skipped_count}")
    logger.info(f"❌ Failed: {failed_count}")
    logger.info(f"📁 Destination: {config.DESTINATION_DIR}")
    logger.info("=" * 60)
    
    # Log failed downloads
    failed_blobs = [r for r in results if not r["success"]]
    if failed_blobs:
        logger.error("\n--- FAILED DOWNLOADS ---")
        for result in failed_blobs:
            logger.error(f"{result['blob']}: {result['message']}")
        logger.error("=" * 60)
    
    logger.info("✅ Download pipeline complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Download interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
