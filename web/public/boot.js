// If the app's main script fails to load (a cached index.html naming a
// bundle that a redeploy replaced), reload once to fetch the current page.
window.addEventListener('error', function (e) {
  var t = e && e.target
  if (t && t.tagName === 'SCRIPT' && t.type === 'module') {
    try {
      if (!sessionStorage.getItem('fplabs.reloaded')) {
        sessionStorage.setItem('fplabs.reloaded', '1')
        location.reload()
      }
    } catch (x) { /* storage blocked: nothing to do */ }
  }
}, true)
