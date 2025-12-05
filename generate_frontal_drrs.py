"""
Generate Frontal PA Chest X-ray DRRs from CT Volumes
Fixed to produce proper frontal views instead of lateral views
"""

import os
import io
import time
import numpy as np
import SimpleITK as sitk
import shutil
import pydicom
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from azure.storage.blob import ContainerClient

# -----------------------------
# CONFIGURATION
# -----------------------------
BLOB_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=spartis9488473038;"
    "AccountKey=WxiLwTEm+WEut0AIFRTLiWcXgHhDixXtYtF5gbbGIKLMWANt5wHOVwg/"
    "QzRgz2uG1CHcazDil58i+ASttN+yaA==;EndpointSuffix=core.windows.net"
)
CONTAINER_NAME = "ct-big-data"
INPUT_DIR_IN_DATASTORE = "rsna/rsna_extracted/train/"
OUTPUT_DIR_IN_DATASTORE = "rsna_drrs_and_nifti/"

MAX_WORKERS = 4
TEMP_DOWNLOAD_DIR = "/tmp/drr_processing"

# CT processing parameters
WINDOW_CENTER, WINDOW_WIDTH = 400, 1300
IMAGE_SIZE = [512, 512]

# Configure pydicom to ignore validation errors
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

# -----------------------------
# ROBUST DICOM READER
# -----------------------------
def read_dicom_series_robust(directory):
    """
    Manually reads DICOM files using pydicom, sorts them by Z-position,
    applies rescale slope/intercept, and converts to SimpleITK image.
    Bypasses strict GDCM validation.
    """
    try:
        # 1. List and read all files
        slices = []
        for fname in os.listdir(directory):
            if fname.endswith('.dcm'):
                fpath = os.path.join(directory, fname)
                try:
                    # force=True allows reading files without standard preambles
                    ds = pydicom.dcmread(fpath, force=True)
                    
                    # We need ImagePositionPatient to sort slices
                    if hasattr(ds, 'ImagePositionPatient'):
                        slices.append(ds)
                except:
                    continue

        if not slices:
            return None

        # 2. Sort slices by Z position (ImagePositionPatient[2])
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # 3. Extract pixel data and apply Hounsfield Unit (HU) conversion
        image_volume = []
        for s in slices:
            slope = getattr(s, 'RescaleSlope', 1.0)
            intercept = getattr(s, 'RescaleIntercept', 0.0)
            pixels = s.pixel_array.astype(np.float32)
            pixels = (pixels * slope) + intercept
            image_volume.append(pixels)

        # Stack into 3D numpy array (z, y, x)
        np_volume = np.array(image_volume)

        # 4. Convert to SimpleITK Image
        sitk_image = sitk.GetImageFromArray(np_volume)

        # 5. Set Spatial Metadata from the first slice
        first_slice = slices[0]
        
        # Origin
        origin = [float(x) for x in first_slice.ImagePositionPatient]
        sitk_image.SetOrigin(origin)

        # Spacing
        pixel_spacing = [float(x) for x in first_slice.PixelSpacing]
        if len(slices) > 1:
            z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        else:
            z_spacing = getattr(first_slice, 'SliceThickness', 1.0)
        
        spacing = [pixel_spacing[0], pixel_spacing[1], float(z_spacing)]
        sitk_image.SetSpacing(spacing)

        # Direction (ImageOrientationPatient)
        if hasattr(first_slice, 'ImageOrientationPatient'):
            iop = [float(x) for x in first_slice.ImageOrientationPatient]
            row_cosines = np.array(iop[:3])
            col_cosines = np.array(iop[3:])
            slice_cosines = np.cross(row_cosines, col_cosines)
            direction = list(iop) + list(slice_cosines)
            sitk_image.SetDirection(direction)

        return sitk_image

    except Exception as e:
        print(f"Robust reader error: {e}")
        return None

# -----------------------------
# PROCESSING LOGIC
# -----------------------------
def preprocess_ct(ct_image):
    """Preprocess CT with resampling and bone windowing."""
    try:
        size = ct_image.GetSize()
        spacing = ct_image.GetSpacing()
        
        # Sanity check on spacing
        if any(s <= 0 for s in spacing):
            spacing = (1.0, 1.0, 1.0)
            ct_image.SetSpacing(spacing)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([2.0, 2.0, 2.0])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(ct_image.GetDirection())
        resampler.SetOutputOrigin(ct_image.GetOrigin())
        
        new_size = [int(s * sp / 2.0) for s, sp in zip(size, spacing)]
        if any(sz == 0 for sz in new_size): 
            return None
        resampler.SetSize(new_size)

        ct_image = resampler.Execute(ct_image)

        ct_array = sitk.GetArrayFromImage(ct_image)
        ct_array = np.clip(
            ct_array,
            WINDOW_CENTER - WINDOW_WIDTH/2,
            WINDOW_CENTER + WINDOW_WIDTH/2
        )
        ct_array = (ct_array - (WINDOW_CENTER - WINDOW_WIDTH/2)) / WINDOW_WIDTH
        
        processed_image = sitk.GetImageFromArray(ct_array)
        processed_image.CopyInformation(ct_image)
        return processed_image
    except Exception as e:
        return None

def generate_drr(ct_image, view_angle):
    """
    Generate DRR from CT image.
    view_angle: 0 = PA (frontal chest view)
    
    For FRONTAL PA chest X-ray:
    - Project along anterior-posterior direction
    - Result shows chest from the front (like a real chest X-ray)
    """
    try:
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        # SimpleITK GetArrayFromImage returns: [Z, Y, X]
        # Z = superior-inferior (head to feet)
        # Y = anterior-posterior (front to back)  
        # X = left-right
        
        # For frontal chest X-ray, we want to look at the patient from the FRONT
        # This means projecting along the Y-axis (depth/anterior-posterior)
        
        if view_angle == 0:  # PA frontal view
            # Sum along axis=1 (Y-axis) to get frontal projection
            drr = np.sum(ct_array, axis=1)  # Result: [Z, X] = [height, width]
            
            # Average instead of sum for better dynamic range
            drr = drr / ct_array.shape[1]
            
            # Flip horizontally to match radiological convention (patient's right on image left)
            drr = np.fliplr(drr)
            
        else:
            return None

        # Normalize to [0, 1] - same as reference script
        drr_min = np.min(drr)
        drr_max = np.max(drr)
        drr = (drr - drr_min) / (drr_max - drr_min + 1e-6)
        
        # Add scatter and clip - matches reference script exactly
        drr = np.clip(drr + 0.05, 0, 1)
        
        # Convert to 8-bit image - SimpleITK method from reference
        drr_sitk_temp = sitk.GetImageFromArray(drr)
        drr_sitk = sitk.Cast(sitk.RescaleIntensity(drr_sitk_temp, 0, 255), sitk.sitkUInt8)
        return drr_sitk
    except Exception as e:
        print(f"DRR generation error: {e}")
        return None

def process_and_upload_patient(patient_id):
    """Process a single patient: download DICOM, generate frontal DRR, upload."""
    local_patient_dir = os.path.join(TEMP_DOWNLOAD_DIR, patient_id)
    local_nifti_path = ""
    
    try:
        # Disk check
        total, used, free = shutil.disk_usage(TEMP_DOWNLOAD_DIR)
        if (free / total) < 0.1:
            time.sleep(30)

        os.makedirs(local_patient_dir, exist_ok=True)
        
        # 1. Download DICOM files
        patient_blob_prefix = f"{INPUT_DIR_IN_DATASTORE}{patient_id}/"
        blob_list = container_client.list_blobs(name_starts_with=patient_blob_prefix)
        
        dcm_files_downloaded = 0
        for blob in blob_list:
            if blob.name.endswith('.dcm'):
                local_file_path = os.path.join(local_patient_dir, os.path.basename(blob.name))
                blob_client = container_client.get_blob_client(blob.name)
                with open(local_file_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                dcm_files_downloaded += 1
        
        if dcm_files_downloaded == 0:
            return f"⚠️ No .dcm files found for {patient_id}"

        # 2. Read CT Image (with robust fallback)
        ct_image = None
        method = "Standard"
        
        # Attempt 1: Standard SimpleITK
        try:
            series_reader = sitk.ImageSeriesReader()
            dicom_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(local_patient_dir)
            if dicom_names:
                series_reader.SetFileNames(dicom_names)
                ct_image = series_reader.Execute()
        except Exception:
            pass

        # Attempt 2: Robust Pydicom Reader
        if ct_image is None:
            ct_image = read_dicom_series_robust(local_patient_dir)
            method = "Robust"

        if ct_image is None:
            return f"⚠️ Failed to read series for {patient_id} (Standard & Robust methods failed)"

        # 3. Preprocess CT
        processed_ct = preprocess_ct(ct_image)
        if processed_ct is None: 
            return f"⚠️ Preprocessing failed {patient_id}"
        
        # 4. Generate PA Frontal DRR
        pa_drr = generate_drr(processed_ct, 0)
        if pa_drr is None: 
            return f"⚠️ PA DRR generation failed {patient_id}"

        # 5. Resize and save to buffer
        pa_filename = f"{patient_id}_pa_drr.png"
        pa_buffer = BytesIO()
        pa_array = sitk.GetArrayFromImage(pa_drr)
        pil_image_pa = Image.fromarray(pa_array).resize(tuple(IMAGE_SIZE), Image.Resampling.LANCZOS)
        pil_image_pa.save(pa_buffer, format="PNG")
        pa_buffer.seek(0)

        # 6. Upload PA DRR to Azure
        pa_drr_dest = os.path.join(OUTPUT_DIR_IN_DATASTORE, patient_id, pa_filename)
        container_client.upload_blob(pa_drr_dest, pa_buffer, overwrite=True)

        return f"✅ Processed {patient_id} [PA Frontal] [{method}]"

    except Exception as e:
        return f"❌ Error {patient_id}: {e}"
    finally:
        if os.path.exists(local_patient_dir): 
            shutil.rmtree(local_patient_dir)
        if local_nifti_path and os.path.exists(local_nifti_path): 
            os.remove(local_nifti_path)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print(f"🔹 Connecting to container '{CONTAINER_NAME}'...")
    try:
        container_client = ContainerClient.from_connection_string(BLOB_CONNECTION_STRING, CONTAINER_NAME)
        print("✅ Connected.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        exit()

    if os.path.exists(TEMP_DOWNLOAD_DIR): 
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

    print("\n🔎 Listing source patients...")
    all_patient_ids = set()
    for blob in container_client.list_blobs(name_starts_with=INPUT_DIR_IN_DATASTORE):
        parts = blob.name.split('/')
        if len(parts) > 3: 
            all_patient_ids.add(parts[3])

    print("\n🔎 Checking existing outputs...")
    existing_ids = set()
    for blob in container_client.list_blobs(name_starts_with=OUTPUT_DIR_IN_DATASTORE):
        parts = blob.name.split('/')
        if len(parts) > 2: 
            existing_ids.add(parts[1])

    # Process ALL patients to regenerate frontal DRRs
    to_process = sorted(list(all_patient_ids))
    print(f"🚀 Starting {len(to_process)} patients total.")
    print(f"   {len(existing_ids)} folders exist and will have PA DRRs regenerated with FRONTAL view...\n")

    if to_process:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for res in executor.map(process_and_upload_patient, to_process):
                print(res)
    else:
        print("⚠️ Nothing to process.")

    print("\n✅ DRR generation complete!")
