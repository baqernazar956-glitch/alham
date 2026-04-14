document.addEventListener('DOMContentLoaded', function () {
  const search = document.getElementById('genreSearch');
  const grid = document.getElementById('genresGrid');
  const selectAll = document.getElementById('btnSelectAll');
  const clearAll = document.getElementById('btnClearAll');
  if (search && grid) {
    search.addEventListener('input', () => {
      const q = search.value.toLowerCase();
      grid.querySelectorAll('.genre-item').forEach(item => {
        const name = item.textContent.trim().toLowerCase();
        item.style.display = name.includes(q) ? '' : 'none';
      });
    });
  }

  if (selectAll && grid) {
    selectAll.addEventListener('click', () => {
      grid.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
    });
  }

  if (clearAll && grid) {
    clearAll.addEventListener('click', () => {
      grid.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
    });
  }
});
