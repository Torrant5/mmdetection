#!/usr/bin/env python
"""
[DEPRECATED - DO NOT USE] This script incorrectly changes COCO standard IDs.

mmdetection's CocoDataset automatically handles the mapping:
- COCO standard IDs: person=1, sports ball=33
- Internal labels: person=0, sports ball=1 (via cat2label mapping)

KEEP COCO STANDARD IDs for better transfer learning.
This script is kept only for reference. Use verify_category_mapping.py instead.
"""

import json
import os
import shutil
from datetime import datetime


def fix_category_ids_to_sequential(json_path, backup=True):
    """
    Fix category IDs to sequential (1, 2) for 2-class model.
    
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
        backup_path = json_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2(json_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # WARNING: This incorrectly changes COCO standard IDs
    # Should keep: person=1, sports ball=33
    category_mapping = {}
    for cat in data['categories']:
        if cat['name'] == 'sports ball' and cat['id'] == 33:
            # WARNING: Should NOT change from 33
            old_id = cat['id']
            cat['id'] = 2  # INCORRECT: Breaking COCO standard
            category_mapping[old_id] = 2
            print(f"  WARNING: Incorrectly changing sports ball from {old_id} to 2")
        elif cat['name'] == 'person' and cat['id'] != 1:
            old_id = cat['id']
            cat['id'] = 1
            category_mapping[old_id] = 1
            print(f"  Changed category ID: person from {old_id} to 1")
    
    # Update annotations with new category IDs
    if category_mapping:
        updated_count = 0
        for ann in data['annotations']:
            if ann['category_id'] in category_mapping:
                old_id = ann['category_id']
                ann['category_id'] = category_mapping[old_id]
                updated_count += 1
        print(f"  Updated {updated_count} annotations with new category IDs")
    
    # Verify the result
    final_cats = {cat['id']: cat['name'] for cat in data['categories']}
    print(f"  Final categories: {final_cats}")
    
    # Count annotations per category
    cat_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    print(f"  Annotations per category: {cat_counts}")
    
    # Save the modified JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved modified file: {json_path}")
    
    return category_mapping


def main():
    """Main function to fix category IDs in train and test annotations."""
    base_path = "data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images2"
    
    # Files to process
    files_to_fix = [
        os.path.join(base_path, "train", "instances_train.json"),
        os.path.join(base_path, "test", "instances_test.json"),
    ]
    
    print("Fixing category IDs to sequential (1, 2) for 2-class model...")
    print("=" * 60)
    
    for json_file in files_to_fix:
        if os.path.exists(json_file):
            fix_category_ids_to_sequential(json_file)
        else:
            print(f"Warning: File not found: {json_file}")
        print()
    
    print("=" * 60)
    print("Category ID fixing complete!")
    print("\nNote: Backup files have been created with .backup_* extension")
    print("If you need to revert, simply rename the backup files.")
    print("\nIMPORTANT: After this fix:")
    print("  - person: category_id 1 (model class 0)")
    print("  - sports ball: category_id 2 (model class 1)")


if __name__ == "__main__":
    main()