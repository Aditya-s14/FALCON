#!/usr/bin/env python3
"""
Step 2: Collecting Location-Specific Bird Audio from Xeno-canto
Target region: India (West Bengal / Sundarbans area)
Bounding box: 21.5504,88.2518,22.2017,89.0905

IMPORTANT: Xeno-canto API v3 requires an API key!
Get your key at: https://xeno-canto.org/account (free registration)
"""

import os
import json
import shutil
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================
# Get your API key from: https://xeno-canto.org/account
API_KEY = os.environ.get("XENO_CANTO_API_KEY", "YOUR_API_KEY_HERE")

# Geographic search parameters
COUNTRY = "India"
# Bounding box: lat_min, lat_max, lon_min, lon_max (Sundarbans/West Bengal area)
BOX = "21.5504,22.2017,88.2518,89.0905"

# Output directories
OUTPUT_DIR = "output/xeno_canto"
ORGANIZED_DIR = "output/organized_audio"

# Download settings
MAX_WORKERS = 4  # Number of parallel downloads
MAX_RECORDINGS = 100  # Maximum recordings to download (set to None for all)
# ============================================================

def check_api_key():
    """Check if API key is configured."""
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        print("=" * 60)
        print("ERROR: Xeno-canto API key not configured!")
        print("=" * 60)
        print()
        print("Xeno-canto API v3 requires an API key for access.")
        print()
        print("To get your free API key:")
        print("  1. Visit: https://xeno-canto.org/account")
        print("  2. Create an account or log in")
        print("  3. Find your API key in your account settings")
        print()
        print("Then either:")
        print("  a) Set environment variable: export XENO_CANTO_API_KEY='your-key'")
        print("  b) Edit this script and replace 'YOUR_API_KEY_HERE' with your key")
        print()
        return False
    return True

def fetch_json(url, retries=5):
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'BirdNET-Finetune/1.0',
                'Accept': 'application/json'
            })
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode('UTF-8'))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)  # Exponential backoff
    return None

def step1_retrieve_metadata():
    """Retrieve metadata from Xeno-canto API v3."""
    print("=" * 60)
    print("Step 2.1 & 2.2: Querying Xeno-canto API v3")
    print("=" * 60)
    print(f"Country: {COUNTRY}")
    print(f"Bounding Box: {BOX}")
    
    # Build query for Xeno-canto API v3
    # Format: cnt:COUNTRY box:LAT_MIN,LAT_MAX,LON_MIN,LON_MAX
    query = f"cnt:{COUNTRY} box:{BOX}"
    base_url = "https://xeno-canto.org/api/3/recordings"
    
    all_recordings = []
    page = 1
    
    print(f"\nQuery: {query}")
    print("Fetching metadata pages...")
    
    while True:
        url = f"{base_url}?query={urllib.request.quote(query)}&key={API_KEY}&page={page}"
        print(f"  Fetching page {page}...")
        
        data = fetch_json(url)
        
        if not data:
            print(f"  Failed to fetch page {page}")
            break
        
        # Check for API errors
        if 'error' in data:
            print(f"  API Error: {data.get('message', 'Unknown error')}")
            break
            
        if 'recordings' not in data or not data['recordings']:
            if page == 1:
                print("  No recordings found for this query.")
            break
            
        recordings = data['recordings']
        all_recordings.extend(recordings)
        
        num_pages = int(data.get('numPages', 1))
        num_total = int(data.get('numRecordings', 0))
        print(f"  Got {len(recordings)} recordings (page {page}/{num_pages}, total: {num_total})")
        
        # Check if we've hit our limit
        if MAX_RECORDINGS and len(all_recordings) >= MAX_RECORDINGS:
            all_recordings = all_recordings[:MAX_RECORDINGS]
            print(f"  Reached maximum limit of {MAX_RECORDINGS} recordings")
            break
        
        if page >= num_pages:
            break
        page += 1
        time.sleep(0.5)  # Be nice to the API
    
    print(f"\nTotal recordings to process: {len(all_recordings)}")
    
    if all_recordings:
        # Analyze species found
        species_set = set()
        for rec in all_recordings:
            gen = rec.get('gen', '')
            sp = rec.get('sp', '')
            en = rec.get('en', '')
            if gen and sp:
                species_set.add(f"{gen} {sp} ({en})")
        
        print(f"Unique species: {len(species_set)}")
        print("\nSpecies found:")
        for i, sp in enumerate(sorted(species_set)):
            print(f"  {i+1}. {sp}")
    
    return all_recordings

def download_recording(rec, output_dir):
    """Download a single recording."""
    try:
        rec_id = rec.get('id', '')
        file_url = rec.get('file', '')
        
        if not file_url or not rec_id:
            return None
        
        # Handle relative URLs
        if file_url.startswith('//'):
            file_url = 'https:' + file_url
        elif not file_url.startswith('http'):
            file_url = 'https://xeno-canto.org' + file_url
            
        # Determine file extension
        ext = '.mp3'
        if 'wav' in file_url.lower():
            ext = '.wav'
            
        filename = f"XC{rec_id}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return filepath
            
        # Download the file
        req = urllib.request.Request(file_url, headers={'User-Agent': 'BirdNET-Finetune/1.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        return filepath
    except Exception as e:
        print(f"  Error downloading {rec.get('id', 'unknown')}: {e}")
        return None

def step2_download_recordings(recordings):
    """Download all recordings with parallel processing."""
    print("\n" + "=" * 60)
    print("Step 2.3: Downloading Audio Recordings")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Recordings to download: {len(recordings)}")
    print(f"Using {MAX_WORKERS} parallel workers")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_recording, rec, OUTPUT_DIR): rec for rec in recordings}
        
        for i, future in enumerate(as_completed(futures), 1):
            filepath = os.path.join(OUTPUT_DIR, f"XC{futures[future].get('id', '')}.mp3")
            if os.path.exists(filepath) and future.result() == filepath:
                # Check if it was already there
                skipped += 1
            
            result = future.result()
            if result:
                downloaded += 1
            else:
                failed += 1
            
            if i % 10 == 0 or i == len(recordings):
                print(f"  Progress: {i}/{len(recordings)} ({downloaded} downloaded, {failed} failed)")
    
    print(f"\nDownload complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    
    return downloaded

def step3_organize_by_species(recordings):
    """Organize downloaded files by species."""
    print("\n" + "=" * 60)
    print("Step 2.4: Organizing Audio Files by Species")
    print("=" * 60)
    
    os.makedirs(ORGANIZED_DIR, exist_ok=True)
    
    organized_count = 0
    species_counts = {}
    
    for rec in recordings:
        try:
            # Get species info
            genus = rec.get('gen', 'Unknown')
            species = rec.get('sp', 'unknown')
            common_name = rec.get('en', 'Unknown')
            rec_id = rec.get('id', '')
            
            # Create folder name: "Genus species_CommonName"
            folder_name = f"{genus} {species}_{common_name}"
            # Clean folder name (remove invalid characters)
            folder_name = "".join(c for c in folder_name if c.isalnum() or c in " _-").strip()
            
            species_dir = os.path.join(ORGANIZED_DIR, folder_name)
            os.makedirs(species_dir, exist_ok=True)
            
            # Find the downloaded file
            source_patterns = [
                os.path.join(OUTPUT_DIR, f"XC{rec_id}.mp3"),
                os.path.join(OUTPUT_DIR, f"XC{rec_id}.wav"),
            ]
            
            for source_file in source_patterns:
                if os.path.exists(source_file):
                    dest_file = os.path.join(species_dir, os.path.basename(source_file))
                    if not os.path.exists(dest_file):
                        shutil.copy2(source_file, dest_file)
                        organized_count += 1
                        species_counts[folder_name] = species_counts.get(folder_name, 0) + 1
                    break
                    
        except Exception as e:
            print(f"Error organizing recording {rec.get('id', 'unknown')}: {e}")
    
    print(f"\nOrganized {organized_count} files into {len(species_counts)} species folders")
    print(f"Organized directory: {ORGANIZED_DIR}")
    
    if species_counts:
        print("\nRecordings per species:")
        for sp, count in sorted(species_counts.items(), key=lambda x: -x[1]):
            print(f"  {sp}: {count} recordings")

def main():
    print("=" * 60)
    print("STEP 2: COLLECTING LOCATION-SPECIFIC BIRD AUDIO")
    print("=" * 60)
    print()
    
    # Check API key first
    if not check_api_key():
        return
    
    # Step 2.1 & 2.2: Retrieve metadata
    recordings = step1_retrieve_metadata()
    
    if not recordings:
        print("\nNo recordings found for the specified region. Exiting.")
        return
    
    # Save metadata for reference
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(recordings, f, indent=2)
    print(f"\nMetadata saved to: {metadata_file}")
    
    # Step 2.3: Download recordings
    step2_download_recordings(recordings)
    
    # Step 2.4: Organize by species
    step3_organize_by_species(recordings)
    
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE!")
    print("=" * 60)
    print(f"Raw downloads: {OUTPUT_DIR}")
    print(f"Organized by species: {ORGANIZED_DIR}")

if __name__ == "__main__":
    main()
