import os
import sys

# --- Configuration ---
# Base directory
EIA_STACK_DIR = r"D:\EIA_STACK"  # <<< Parent directory

# Specific files to delete from the EIA_STACK_DIR
SPECIFIC_FILES_IN_EIA_STACK = [
    "power_burns_daily.csv",
    "power_burns_monthly.csv",
    "power_burns_hourly.csv",
    "final_power_burns_multi_features.csv"
]

# Regions within EIA_STACK_DIR to clean recursively
# Add all region folder names here that need .parquet and .csv files deleted recursively
REGIONS_TO_CLEAN_RECURSIVELY = [
    "Carolina",
    "Texas",
    "California",
    "Midatlantic",
    "Newyork",
    "Newengland",
    "Southeast",
    "Southwest",
    "Northwest",
    "Midwest",
    "Tennessee",
    "Central",
    "Florida",
    "granular_storage",
    "Pricing_RL"
]

# File extensions to delete recursively within each region in REGIONS_TO_CLEAN_RECURSIVELY
EXTENSIONS_TO_DELETE_IN_REGIONS = ['.parquet', '.csv', '.png']  # Case-insensitive check will be used

# *** NEW: Folder name to exclude from recursive deletion within each region ***
# Any folder with this name (case-insensitive) will be skipped entirely.
EXCLUSION_FOLDER_NAME = "Capacity"
# --- End Configuration ---

def find_specific_files_in_parent(parent_path, specific_filenames_list):
    """Identifies specific named files in the parent_path."""
    found_files = []
    print(f"\n--- Identifying specific files in: {parent_path} ---")
    if not os.path.isdir(parent_path):
        print(f"ERROR: Parent directory for specific files not found: {parent_path}.")
        return found_files  # Return empty list, main will check overall list later
    
    for specific_file_name in specific_filenames_list:
        file_path = os.path.join(parent_path, specific_file_name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            found_files.append(file_path)
            print(f"  Found specific file: {file_path}")
        else:
            print(f"  Specific file NOT found or is not a file: {file_path}")
    if not found_files:
        print(f"  No specified files found in {parent_path}.")
    return found_files

def find_files_recursively_in_region(region_path, extensions_list, exclusion_folder_name):
    """
    Identifies files with specified extensions in region_path AND ALL ITS SUBFOLDERS,
    EXCLUDING any directory matching the exclusion_folder_name.
    """
    found_files_recursive = []
    print(f"\n--- Identifying files in: {region_path} (and its subfolders) ---")
    if not os.path.isdir(region_path):
        print(f"WARNING: Region directory not found: {region_path}. Skipping this region.")
        return found_files_recursive

    print(f"Looking for files with extensions: {', '.join(extensions_list)} (case-insensitive)")
    print(f"Excluding any directory named: '{exclusion_folder_name}'")
    
    exclusion_folder_lower = exclusion_folder_name.lower()

    for root, dirs, files in os.walk(region_path):
        # --- MODIFICATION START ---
        # To prevent os.walk from descending into the exclusion folder, we modify the 'dirs' list in-place.
        # This is more efficient than checking every file's path.
        # We do this case-insensitively.
        dirs_to_remove = [d for d in dirs if d.lower() == exclusion_folder_lower]
        if dirs_to_remove:
            for dir_to_remove in dirs_to_remove:
                # Remove the directory from the list so os.walk won't visit it.
                dirs.remove(dir_to_remove)
                print(f"  Excluding directory from search: {os.path.join(root, dir_to_remove)}")
        # --- MODIFICATION END ---
                
        for file_name in files:
            if any(file_name.lower().endswith(ext.lower()) for ext in extensions_list):
                file_path = os.path.join(root, file_name)
                found_files_recursive.append(file_path)
    
    if found_files_recursive:
        print(f"  Found {len(found_files_recursive)} files in {region_path} (and subfolders) matching criteria.")
    else:
        print(f"  No files found in {region_path} (and subfolders) matching criteria.")
    return found_files_recursive

def process_deletions(master_files_list_to_delete):
    """Handles deletion for the master list of files."""
    if not master_files_list_to_delete:
        print("\nNo files identified for deletion across all criteria.")
        return True  # Considered success as there's nothing to do

    print("\n--- Files identified for deletion (MASTER LIST) ---")
    for file_path in master_files_list_to_delete:
        print(file_path)
    print(f"-----------------------------------------------------------")
    print(f"Total unique files to be deleted: {len(master_files_list_to_delete)}")
    print(f"-----------------------------------------------------------")

    # --- SAFETY CONFIRMATION REMOVED ---
    # The following block has been commented out to bypass user confirmation:
    # confirm = input(f"\nAre you SURE you want to delete {len(master_files_list_to_delete)} file(s) listed above? (yes/NO): ")
    # if confirm.lower() != 'yes':
    #     print("Deletion cancelled by user.")
    #     return False
    # --- END SAFETY CONFIRMATION REMOVED ---

    deleted_count = 0
    error_count = 0
    print("\n--- Starting Deletion Process ---")
    for file_path in master_files_list_to_delete:
        try:
            # !!! DANGER ZONE - os.remove IS NOW ACTIVE !!!
            os.remove(file_path)
            # !!! DANGER ZONE !!!
            print(f"DELETED: {file_path}")
            deleted_count += 1
        except OSError as e:
            print(f"ERROR deleting {file_path}: {e}")
            error_count += 1
        except Exception as e:
            print(f"UNEXPECTED ERROR deleting {file_path}: {e}")
            error_count += 1
    print("-------------------------------")

    print(f"\nSummary:")
    print(f"  Files successfully deleted: {deleted_count}")
    print(f"  Errors during deletion: {error_count}")

    return error_count == 0

if __name__ == "__main__":
    print("="*60)
    print("=== ITERATIVE RECURSIVE FILE DELETION SCRIPT ===")
    print("="*60)
    print("WARNING: This script will permanently delete files.")
    print("It will search RECURSIVELY in the specified region directories.")
    print("`os.remove()` IS ACTIVE. Files will be deleted WITHOUT confirmation.")
    print("Please ensure paths and region list are set correctly and you have backups.")
    print(f"EIA_STACK_DIR (for specific files & as base for regions): {EIA_STACK_DIR}")
    print(f"SPECIFIC FILES to target in '{EIA_STACK_DIR}': {SPECIFIC_FILES_IN_EIA_STACK}")
    print(f"REGIONS for recursive cleaning (subfolders of EIA_STACK_DIR): {REGIONS_TO_CLEAN_RECURSIVELY}")
    print(f"EXTENSIONS TO DELETE in regions (recursively): {', '.join(EXTENSIONS_TO_DELETE_IN_REGIONS)}")
    print(f"EXCLUSION FOLDER NAME (will be skipped): '{EXCLUSION_FOLDER_NAME}'")
    print("="*60)

    if not EIA_STACK_DIR:
        print("\nERROR: EIA_STACK_DIR is not properly set.")
        sys.exit(1)
    
    if not os.path.isdir(EIA_STACK_DIR):
        print(f"\nERROR: The main EIA_STACK_DIR directory '{EIA_STACK_DIR}' does not exist or is not a directory.")
        sys.exit(1)

    master_deletion_candidates = []

    # 1. Get specific files from the parent EIA_STACK_DIR
    specific_parent_files = find_specific_files_in_parent(EIA_STACK_DIR, SPECIFIC_FILES_IN_EIA_STACK)
    master_deletion_candidates.extend(specific_parent_files)

    # 2. Get files from each specified region recursively
    if not REGIONS_TO_CLEAN_RECURSIVELY:
        print("\nNo regions specified in REGIONS_TO_CLEAN_RECURSIVELY. Skipping regional recursive search.")
    else:
        for region_name in REGIONS_TO_CLEAN_RECURSIVELY:
            current_region_path = os.path.join(EIA_STACK_DIR, region_name)
            # Pass the exclusion folder name to the function
            regional_files = find_files_recursively_in_region(
                current_region_path, 
                EXTENSIONS_TO_DELETE_IN_REGIONS,
                EXCLUSION_FOLDER_NAME
            )
            master_deletion_candidates.extend(regional_files)

    # 3. Deduplicate the master list
    if master_deletion_candidates:
        unique_files_to_delete = sorted(list(set(master_deletion_candidates)))
        if len(unique_files_to_delete) < len(master_deletion_candidates):
            print(f"\nNote: Removed {len(master_deletion_candidates) - len(unique_files_to_delete)} duplicate entries from the overall deletion list.")
        master_deletion_candidates = unique_files_to_delete
    
    # 4. Process deletions for the consolidated list
    success = process_deletions(master_deletion_candidates)

    if success:
        print("\nFile deletion process completed successfully (or no files to delete/errors that didn't halt).")
        sys.exit(0)
    else:
        print("\nFile deletion process encountered errors.")
        sys.exit(1)