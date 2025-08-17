#!/usr/bin/env python3
"""
Helper script to update the rep number in commons.py
Usage: python update_rep.py <rep_number>
"""

import sys
import os
import re

def update_rep_in_commons(rep_num):
    """Update the generate_data_path_latest_rep line in commons.py"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    commons_file = os.path.join(project_root, 'pkgs', 'commons.py')
    
    if not os.path.exists(commons_file):
        print(f"Error: commons.py not found at {commons_file}")
        return False
    
    try:
        with open(commons_file, 'r') as f:
            content = f.read()
        
        pattern = r"current_rep = \d+"
        replacement = f"current_rep = {rep_num}"
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content == content:
            print(f"Warning: No changes made. Pattern might not match or rep{rep_num} already set.")
            return True
        
        with open(commons_file, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully updated commons.py to use rep{rep_num}")
        return True
        
    except Exception as e:
        print(f"Error updating commons.py: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_rep.py <rep_number>")
        sys.exit(1)
    
    try:
        rep_num = int(sys.argv[1])
        if not update_rep_in_commons(rep_num):
            sys.exit(1)
    except ValueError:
        print("Error: rep_number must be an integer")
        sys.exit(1)
