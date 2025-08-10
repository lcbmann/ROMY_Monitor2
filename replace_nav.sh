#!/bin/bash

# Script to replace navigation sections with placeholder in all HTML pages

WEBSITE_DIR="/Users/liamb/Documents/GitHub/ROMY_Monitor2/webpage/website"
PAGES=("beam-wander.html" "beat-drift.html" "backscatter.html" "environmental.html" "barometric.html")

for page in "${PAGES[@]}"; do
    echo "Replacing navigation in $page..."
    
    # Use a Python script to replace the navigation section
    python3 << EOF
import re

# Read the file
with open('$WEBSITE_DIR/$page', 'r') as f:
    content = f.read()

# Pattern to match the entire nav section
nav_pattern = r'  <nav class="navbar is-light" role="navigation">.*?</nav>'

# Replacement
replacement = '''  <!-- Navigation placeholder -->
  <div id="navigation-placeholder"></div>'''

# Replace with multiline and dotall flags
new_content = re.sub(nav_pattern, replacement, content, flags=re.DOTALL)

# Write back to file
with open('$WEBSITE_DIR/$page', 'w') as f:
    f.write(new_content)

print(f"Updated $page navigation section")
EOF
    
done

echo "All navigation sections updated!"
