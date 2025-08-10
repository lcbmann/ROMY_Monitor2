#!/bin/bash

# Script to update all HTML pages to use modular navigation

WEBSITE_DIR="/Users/liamb/Documents/GitHub/ROMY_Monitor2/webpage/website"
PAGES=("beam-wander.html" "beat-drift.html" "backscatter.html" "environmental.html" "barometric.html")

for page in "${PAGES[@]}"; do
    echo "Updating $page..."
    
    # Add navigation script to head
    sed -i '' 's|<link rel="stylesheet" href="style.css" />|<link rel="stylesheet" href="style.css" />\
  <script src="js/navigation.js"></script>|' "$WEBSITE_DIR/$page"
    
    echo "Updated $page"
done

echo "All pages updated successfully!"
