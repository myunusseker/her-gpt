#!/usr/bin/env python3
"""
Test script to verify that rendering is identical between GUI and DIRECT modes.
"""

import numpy as np
import os
from src.environments.peg_insertion_environment import PegInsertionEnvironment
from PIL import Image

def compare_images(img1_path, img2_path):
    """Compare two images and return the difference."""
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    
    if img1.shape != img2.shape:
        print(f"Image shapes differ: {img1.shape} vs {img2.shape}")
        return False
    
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max pixel difference: {max_diff}")
    print(f"Mean pixel difference: {mean_diff}")
    
    # Save difference image for inspection
    diff_normalized = (diff / max(max_diff, 1) * 255).astype(np.uint8)
    diff_img_path = img1_path.replace('.png', '_diff.png')
    Image.fromarray(diff_normalized).save(diff_img_path)
    print(f"Difference image saved to: {diff_img_path}")
    
    # Consider images identical if max difference is very small
    return max_diff < 5  # Allow small differences due to floating point precision

def test_rendering_quality_vs_consistency():
    """Test both high-quality OpenGL and consistent software rendering."""
    print("Testing rendering options: Quality vs Consistency")
    print("="*60)
    
    # Create output directory
    os.makedirs("data/peg_insertion_test", exist_ok=True)
    
    # Test 1: High-Quality OpenGL Rendering
    print("\n1. Testing HIGH-QUALITY OPENGL rendering...")
    print("   This gives best visual quality but may have minor differences between GUI/DIRECT")
    
    env_gui_hq = PegInsertionEnvironment(gui=True, hz=60, force_consistent_rendering=False)
    env_gui_hq.reset()
    env_gui_hq.apply_action(offset=np.array([0.01, 0.0, 0.0]))
    env_gui_hq.render_views(
        save_images=True,
        side_img_path="data/peg_insertion_test/side_view_gui_hq.png",
        wrist_img_path="data/peg_insertion_test/wrist_view_gui_hq.png"
    )
    env_gui_hq.disconnect()
    
    env_direct_hq = PegInsertionEnvironment(gui=False, hz=60, force_consistent_rendering=False)
    env_direct_hq.reset()
    env_direct_hq.apply_action(offset=np.array([0.01, 0.0, 0.0]))
    env_direct_hq.render_views(
        save_images=True,
        side_img_path="data/peg_insertion_test/side_view_direct_hq.png",
        wrist_img_path="data/peg_insertion_test/wrist_view_direct_hq.png"
    )
    env_direct_hq.disconnect()
    
    print("\n   Comparing high-quality OpenGL images...")
    hq_wrist_identical = compare_images(
        "data/peg_insertion_test/wrist_view_gui_hq.png",
        "data/peg_insertion_test/wrist_view_direct_hq.png"
    )
    hq_side_identical = compare_images(
        "data/peg_insertion_test/side_view_gui_hq.png",
        "data/peg_insertion_test/side_view_direct_hq.png"
    )
    
    # Test 2: Consistent Software Rendering
    print("\n2. Testing CONSISTENT SOFTWARE rendering...")
    print("   This guarantees pixel-perfect consistency but lower visual quality")
    
    env_gui_consistent = PegInsertionEnvironment(gui=True, hz=60, force_consistent_rendering=True)
    env_gui_consistent.reset()
    env_gui_consistent.apply_action(offset=np.array([0.01, 0.0, 0.0]))
    env_gui_consistent.render_views(
        save_images=True,
        side_img_path="data/peg_insertion_test/side_view_gui_consistent.png",
        wrist_img_path="data/peg_insertion_test/wrist_view_gui_consistent.png"
    )
    env_gui_consistent.disconnect()
    
    env_direct_consistent = PegInsertionEnvironment(gui=False, hz=60, force_consistent_rendering=True)
    env_direct_consistent.reset()
    env_direct_consistent.apply_action(offset=np.array([0.01, 0.0, 0.0]))
    env_direct_consistent.render_views(
        save_images=True,
        side_img_path="data/peg_insertion_test/side_view_direct_consistent.png",
        wrist_img_path="data/peg_insertion_test/wrist_view_direct_consistent.png"
    )
    env_direct_consistent.disconnect()
    
    print("\n   Comparing consistent software images...")
    consistent_wrist_identical = compare_images(
        "data/peg_insertion_test/wrist_view_gui_consistent.png",
        "data/peg_insertion_test/wrist_view_direct_consistent.png"
    )
    consistent_side_identical = compare_images(
        "data/peg_insertion_test/side_view_gui_consistent.png",
        "data/peg_insertion_test/side_view_direct_consistent.png"
    )
    
    # Results
    print("\n" + "="*60)
    print("RENDERING COMPARISON RESULTS:")
    print("="*60)
    
    print("\n📊 HIGH-QUALITY OPENGL RENDERING:")
    print(f"   Wrist view identical: {'✓' if hq_wrist_identical else '✗'}")
    print(f"   Side view identical:  {'✓' if hq_side_identical else '✗'}")
    print("   → Best visual quality, minor pixel differences possible")
    
    print("\n🎯 CONSISTENT SOFTWARE RENDERING:")
    print(f"   Wrist view identical: {'✓' if consistent_wrist_identical else '✗'}")
    print(f"   Side view identical:  {'✓' if consistent_side_identical else '✗'}")
    print("   → Guaranteed consistency, lower visual quality")
    
    print("\n💡 RECOMMENDATION:")
    if hq_wrist_identical and hq_side_identical:
        print("   Use HIGH-QUALITY mode - you get both quality AND consistency! 🎉")
    else:
        print("   Choose based on your needs:")
        print("   • For ML training/consistency: force_consistent_rendering=True")
        print("   • For best visuals/demos: force_consistent_rendering=False")
    
    return hq_wrist_identical and hq_side_identical, consistent_wrist_identical and consistent_side_identical

if __name__ == "__main__":
    test_rendering_quality_vs_consistency()
