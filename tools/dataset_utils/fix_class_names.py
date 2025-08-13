#!/usr/bin/env python
"""
Fix class names in COCO annotations to match CocoDataset expectations.
Changes 'sports_ball' to 'sports ball' (with space).
"""

import json
import os
import shutil
from datetime import datetime


def fix_class_names(json_path, backup=True):
    """
    Fix class names in COCO annotation file.
    
    Args:
        json_path: Path to the annotation JSON file
        backup: Whether to create a backup of the original file
    """
    print(f"Processing: {json_path}")
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create backup if requested
    if backup:
        backup_path = json_path + f'.backup_classname_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(json_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Fix class names
    fixed_count = 0
    for cat in data['categories']:
        if cat['name'] == 'sports_ball':
            cat['name'] = 'sports ball'  # Change to space-separated
            fixed_count += 1
            print("  Changed class name: 'sports_ball' -> 'sports ball'")
    
    # Save the modified JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    if fixed_count > 0:
        print(f"  Saved modified file: {json_path}")
    else:
        print("  No changes needed")
    
    return fixed_count


def main():
    """Main function to fix class names in train and test annotations."""
    base_path = "data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images2"
    
    # Files to process
    files_to_fix = [
        os.path.join(base_path, "train", "instances_train.json"),
        os.path.join(base_path, "test", "instances_test.json"),
    ]
    
    print("Fixing class names to match CocoDataset expectations...")
    print("Changing 'sports_ball' -> 'sports ball'")
    print("=" * 60)
    
    total_fixed = 0
    for json_file in files_to_fix:
        if os.path.exists(json_file):
            fixed = fix_class_names(json_file)
            total_fixed += fixed
        else:
            print(f"Warning: File not found: {json_file}")
        print()
    
    print("=" * 60)
    if total_fixed > 0:
        print(f"✅ Class name fixing complete! Fixed {total_fixed} files.")
        print("\nNote: Backup files have been created with .backup_classname_* extension")
    else:
        print("No changes were needed.")


if __name__ == "__main__":
    main()