# ROMY Monitor Website - Modular Navigation System

## Overview
The website now uses a modular navigation system to reduce code duplication and simplify maintenance. Instead of copying navigation HTML to every page, we load it dynamically from a shared component.

## How It Works

### 1. Navigation Component
- **File**: `components/navigation.html`
- **Purpose**: Contains the complete navigation structure that appears on all pages
- **Benefits**: Single point of truth for navigation - update once, changes apply everywhere

### 2. JavaScript Loader
- **File**: `js/navigation.js`  
- **Purpose**: Loads the navigation component and highlights the current page
- **Features**:
  - Automatically loads navigation on page load
  - Highlights the current page in the navigation
  - Provides fallback if navigation fails to load

### 3. Page Structure
All pages now use this structure:
```html
<!-- Navigation placeholder -->
<div id="navigation-placeholder"></div>
```

Instead of the full `<nav>` section.

## Adding New Pages

### Method 1: Use the Template
1. Copy `templates/page-template.html`
2. Replace placeholders:
   - `[PAGE TITLE]` - Full page title
   - `[PAGE SUBTITLE]` - Hero section subtitle  
   - `[ICON]` - FontAwesome icon name
   - `[SECTION TITLE]` - Main content section title
3. Add your content in the designated areas

### Method 2: Convert Existing Page
1. Add navigation script to `<head>`:
   ```html
   <script src="js/navigation.js"></script>
   ```
2. Replace the entire `<nav>` section with:
   ```html
   <!-- Navigation placeholder -->
   <div id="navigation-placeholder"></div>
   ```

## Updating Navigation

### Adding a New Menu Item
1. Edit `components/navigation.html`
2. Add your new link in the appropriate section
3. Save - changes will appear on all pages automatically

### Changing Menu Structure
1. Edit `components/navigation.html`
2. Modify the dropdown structure or main items as needed
3. Save - all pages update automatically

### Disabling Menu Items
Add the `disabled` class to any menu item:
```html
<a class="navbar-item disabled">
  <i class="fas fa-icon mr-2"></i>Menu Item
</a>
```

## File Structure
```
webpage/website/
├── components/
│   └── navigation.html          # Shared navigation component
├── js/
│   └── navigation.js           # Navigation loader script
├── templates/
│   └── page-template.html      # Template for new pages
├── index.html                  # Homepage
├── seismic_noise.html         # Updated pages...
└── [other pages...]           # All using modular navigation
```

## Benefits

1. **Single Source of Truth**: Navigation changes in one place affect all pages
2. **Consistency**: All pages automatically have identical navigation
3. **Easy Maintenance**: Add new pages without copying navigation code
4. **Quick Updates**: Change menu items, links, or structure instantly across site
5. **Reduced File Size**: Pages are smaller and cleaner
6. **Less Errors**: No risk of navigation inconsistencies between pages

## Current Active Highlighting

The system automatically highlights the current page in the navigation with:
- Darker background color
- Bold font weight
- `is-active` class for Bulma styling

## Browser Compatibility

- Modern browsers that support `fetch()` API
- Graceful fallback displays error message if navigation fails to load
- No external dependencies beyond existing Bulma CSS and FontAwesome

## Future Enhancements

Consider these improvements:
1. Create shared header/footer components
2. Add breadcrumb navigation component  
3. Create page-specific navigation highlighting
4. Add mobile-responsive navigation toggle
