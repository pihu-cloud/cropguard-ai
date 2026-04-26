// ── Navbar scroll effect ──────────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 20) {
    navbar.style.borderBottomColor = 'rgba(74,185,90,.22)';
  } else {
    navbar.style.borderBottomColor = 'rgba(74,185,90,.14)';
  }
}, { passive: true });

// ── Mobile drawer ─────────────────────────────────────────
const burger  = document.getElementById('nav-burger');
const drawer  = document.getElementById('mobile-drawer');
const overlay = document.getElementById('drawer-overlay');

function openDrawer() {
  drawer.classList.add('open');
  overlay.classList.add('open');
  burger.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeDrawer() {
  drawer.classList.remove('open');
  overlay.classList.remove('open');
  burger.classList.remove('open');
  document.body.style.overflow = '';
}
if (burger) burger.addEventListener('click', () => {
  drawer.classList.contains('open') ? closeDrawer() : openDrawer();
});
if (overlay) overlay.addEventListener('click', closeDrawer);

// ── Intersection Observer for animate-up ─────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(el => {
    if (el.isIntersecting) {
      el.target.style.animationPlayState = 'running';
      observer.unobserve(el.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.animate-up').forEach(el => {
  el.style.animationPlayState = 'paused';
  observer.observe(el);
});
