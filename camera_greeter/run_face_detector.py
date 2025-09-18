import cv2
import numpy as np
import time
import os
import argparse
from collections import deque

def main():
    """Main function to select and run the appropriate face detector"""
    parser = argparse.ArgumentParser(description='Face Detection Options')
    parser.add_argument('--mode', type=str, default='fast', 
                        choices=['fast', 'landmarks', 'smile'],
                        help='Detection mode: "fast" uses YOLOv8 with GPU, "landmarks" uses specialized face model, "smile" detects smiles')
    parser.add_argument('--width', type=int, default=640, 
                        help='Camera width resolution')
    parser.add_argument('--height', type=int, default=480, 
                        help='Camera height resolution')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Target camera FPS')
    parser.add_argument('--smile_threshold', type=float, default=0.6,
                        help='Threshold for smile detection (0.0-1.0)')
    args = parser.parse_args()
    
    if args.mode == 'fast':
        # Use the fast GPU-accelerated detector
        print("Starting fast GPU-accelerated face detector...")
        from yolo_face_detector import main as fast_detector
        fast_detector()
    elif args.mode == 'smile':
        # Use the smile detector
        print("Starting face detector with smile recognition...")
        import sys
        sys.argv = [sys.argv[0]]  # Clear args for the next parser
        if args.smile_threshold != 0.4:  # Only add if non-default
            sys.argv.extend(['--smile_threshold', str(args.smile_threshold)])
        from yolo_smile_detector import main as smile_detector
        smile_detector()
    else:
        # Use the specialized face landmarks detector
        print("Starting specialized face landmarks detector...")
        from yolo_face_landmarks_detector import main as landmarks_detector
        landmarks_detector()

if __name__ == "__main__":
    main()