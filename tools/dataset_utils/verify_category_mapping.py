#!/usr/bin/env python
"""
Verify that category IDs in 2-class annotations match COCO standards.
"""

import json
import sys


def verify_annotations(json_path):
    """Verify category IDs match COCO standards."""
    
    # Expected COCO standard mappings
    EXPECTED = {
        'person': 1,
        'sports_ball': 33
    }
    
    print(f"Verifying: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check categories
    issues = []
    for cat in data['categories']:
        cat_name = cat['name']
        cat_id = cat['id']
        
        if cat_name in EXPECTED:
            expected_id = EXPECTED[cat_name]
            if cat_id != expected_id:
                issues.append(f"  ❌ {cat_name}: ID is {cat_id}, expected {expected_id}")
            else:
                print(f"  ✓ {cat_name}: ID {cat_id} (correct)")
        else:
            print(f"  ? Unknown category: {cat_name} with ID {cat_id}")
    
    # Check annotation distribution
    cat_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    print("  Annotation counts by category ID:")
    for cat_id, count in sorted(cat_counts.items()):
        cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), 'unknown')
        print(f"    ID {cat_id} ({cat_name}): {count} annotations")
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("  ✅ All category IDs match COCO standards!")
        return True


def main():
    """Verify both train and test annotation files."""
    base_path = "data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images2"
    
    files_to_check = [
        f"{base_path}/train/instances_train.json",
        f"{base_path}/test/instances_test.json"
    ]
    
    print("Verifying COCO category ID standards for 2-class annotations")
    print("=" * 60)
    
    all_ok = True
    for json_file in files_to_check:
        ok = verify_annotations(json_file)
        all_ok = all_ok and ok
        print()
    
    print("=" * 60)
    if all_ok:
        print("✅ SUCCESS: All files use correct COCO standard IDs!")
        print("\nBenefits:")
        print("- Pre-trained model can leverage sports_ball knowledge")
        print("- No ID confusion between training and inference")
        sys.exit(0)
    else:
        print("❌ FAILED: Some files have incorrect category IDs")
        print("\nRecommendation: Run fix_category_ids.py to correct them")
        sys.exit(1)


if __name__ == "__main__":
    main()