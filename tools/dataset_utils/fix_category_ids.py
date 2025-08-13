#!/usr/bin/env python
"""
Fix category IDs in 2-class COCO annotations to sequential IDs (1, 2).
Changes sports ball from ID 33 to ID 2 for 2-class model compatibility.

This is necessary because:
- Model outputs: class 0 (person), class 1 (sports_ball)
- COCO evaluation expects: category_id = class_index + 1
- So we need: person ID 1, sports_ball ID 2
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
    
    # Fix category IDs: person=1, sports_ball=2
    category_mapping = {}
    for cat in data['categories']:
        if cat['name'] == 'sports ball' and cat['id'] == 33:
            old_id = cat['id']
            cat['id'] = 2  # Sequential ID for 2-class model
            category_mapping[old_id] = 2
            print(f"  Changed category ID: sports ball from {old_id} to 2")
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