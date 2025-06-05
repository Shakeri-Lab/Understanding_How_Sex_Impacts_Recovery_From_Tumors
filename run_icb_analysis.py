#!/usr/bin/env python
"""
Run ICB Analysis
This script runs the ICB analysis pipeline
"""

import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the ICB analysis module
from icb_analysis.icb_main import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ICB Analysis')
    parser.add_argument('--by-type', action='store_true',
                        help='Analyze by ICB type')
    parser.add_argument('--duration', action='store_true',
                        help='Analyze by ICB duration')
    parser.add_argument('--propensity-matching', action='store_true',
                        help='Use propensity score matching')
    
    # Pass the command line arguments to the main function
    args = parser.parse_args()
    
    # Set sys.argv to include only the script name and the base-path
    # This ensures the icb_main.py script receives the correct base path
    sys.argv = [sys.argv[0], '--base-path', '/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors/data']
    
    # Add optional arguments if specified
    if args.by_type:
        sys.argv.append('--by-type')
    if args.duration:
        sys.argv.append('--duration')
    if args.propensity_matching:
        sys.argv.append('--propensity-matching')
    
    # Run the main function
    main() 