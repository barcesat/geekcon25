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
                        choices=['fast', 'landmarks', 'smile', 'serial_smile'],
                        help='Detection mode: "fast" uses YOLOv8 with GPU, "landmarks" uses specialized face model, "smile" detects smiles, "serial_smile" detects smiles and sends serial signal')
    parser.add_argument('--width', type=int, default=640, 
                        help='Camera width resolution')
    parser.add_argument('--height', type=int, default=480, 
                        help='Camera height resolution')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Target camera FPS')
    parser.add_argument('--smile_threshold', type=float, default=0.6,
                        help='Threshold for smile detection (0.0-1.0)')
    parser.add_argument('--smile_duration', type=float, default=5.0,
                        help='Duration in seconds for smile detection to trigger signal')
    parser.add_argument('--serial_port', type=str, default='COM4',
                        help='Serial port to connect to (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)')
    parser.add_argument('--baud_rate', type=int, default=9600,
                        help='Baud rate for serial communication')
    parser.add_argument('--list_ports', action='store_true',
                        help='List available serial ports and exit')
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
        if args.smile_threshold != 0.6:  # Only add if non-default
            sys.argv.extend(['--smile_threshold', str(args.smile_threshold)])
        from yolo_smile_detector import main as smile_detector
        smile_detector()
    elif args.mode == 'serial_smile':
        # Use the serial smile detector
        print("Starting face detector with smile recognition and serial signaling...")
        import sys
        # Clear args for the next parser and set up required ones
        sys.argv = [sys.argv[0]]
        
        # Add non-default arguments
        if args.smile_threshold != 0.6:
            sys.argv.extend(['--smile_threshold', str(args.smile_threshold)])
        if args.smile_duration != 5.0:
            sys.argv.extend(['--smile_duration', str(args.smile_duration)])
        if args.serial_port:
            sys.argv.extend(['--serial_port', args.serial_port])
        if args.baud_rate != 9600:
            sys.argv.extend(['--baud_rate', str(args.baud_rate)])
        if args.list_ports:
            sys.argv.extend(['--list_ports'])
        
        # Import and run the serial smile detector
        from yolo_serial_smile_detector import main as serial_smile_detector
        serial_smile_detector()
    else:
        # Use the specialized face landmarks detector
        print("Starting specialized face landmarks detector...")
        from yolo_face_landmarks_detector import main as landmarks_detector
        landmarks_detector()

if __name__ == "__main__":
    main()