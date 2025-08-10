// Navigation loader for ROMY Monitor
document.addEventListener('DOMContentLoaded', function() {
  // Load navigation component
  fetch('components/navigation.html')
    .then(response => response.text())
    .then(data => {
      const navPlaceholder = document.getElementById('navigation-placeholder');
      if (navPlaceholder) {
        navPlaceholder.innerHTML = data;
        
        // Highlight current page in navigation
        highlightCurrentPage();
      }
    })
    .catch(error => {
      console.error('Error loading navigation:', error);
      // Fallback: show a basic navigation message
      const navPlaceholder = document.getElementById('navigation-placeholder');
      if (navPlaceholder) {
        navPlaceholder.innerHTML = '<p>Navigation could not be loaded</p>';
      }
    });
});

function highlightCurrentPage() {
  const currentPage = window.location.pathname.split('/').pop();
  const navItems = document.querySelectorAll('.navbar-item[href]');
  
  navItems.forEach(item => {
    const href = item.getAttribute('href');
    if (href === currentPage || 
        (currentPage === '' && href === 'index.html') ||
        (currentPage === 'index.html' && href === 'index.html')) {
      item.classList.add('is-active');
      item.style.backgroundColor = 'rgba(0, 0, 0, 0.1)';
      item.style.fontWeight = 'bold';
    }
  });
}
