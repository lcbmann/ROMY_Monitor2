#!/bin/bash
# Script to update website navigation

cd /home/cbuergmann/romy_dev/ROMY_Monitor2/webpage/website

for file in backscatter.html barometric.html beam-wander.html beat-drift.html seismic_noise.html; do
    echo "Updating $file..."
    
    # Replace Main dropdown with simple Home link
    sed -i '
    /<div class="navbar-item has-dropdown is-hoverable">/,/<\/div>/ {
        /<a class="navbar-link">Main<\/a>/ {
            N
            N
            N
            N
            N
            N
            s/.*/<a class="navbar-item" href="index.html">\n            <i class="fas fa-home mr-2"><\/i>Home\n          <\/a>/
        }
    }' "$file"
    
    # Update helicorder.html references to seismic_noise.html
    sed -i 's/helicorder\.html/seismic_noise.html/g' "$file"
    
    echo "Updated $file"
done

echo "All files updated"
