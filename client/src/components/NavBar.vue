<script setup lang="ts">
/**
 * NavBar — Fixed Navigation with Frosted Glass Effect
 * Transparent at top, blurs on scroll. Theme toggle. Mobile menu.
 */
import { ref, onMounted, onUnmounted } from 'vue'
import { useTheme } from '../composables/useTheme'

const { theme, toggleTheme } = useTheme()
const scrolled = ref(false)
const mobileOpen = ref(false)

const navLinks = [
  { label: 'Vision', href: '#vision' },
  { label: 'Metrics', href: '#metrics' },
  { label: 'Journey', href: '#journey' },
  { label: 'Stack', href: '#stack' },
]

function onScroll() {
  scrolled.value = window.scrollY > 40
}

function scrollTo(href: string) {
  mobileOpen.value = false
  const el = document.querySelector(href)
  el?.scrollIntoView({ behavior: 'smooth' })
}

onMounted(() => window.addEventListener('scroll', onScroll, { passive: true }))
onUnmounted(() => window.removeEventListener('scroll', onScroll))
</script>

<template>
  <nav class="navbar" :class="{ 'navbar--scrolled': scrolled }">
    <div class="navbar__inner container">
      <!-- Logo -->
      <a href="#" class="navbar__logo" @click.prevent="window.scrollTo({ top: 0, behavior: 'smooth' })">
        <span class="navbar__logo-v">V</span><span class="navbar__logo-text">YREX</span>
      </a>

      <!-- Desktop Links -->
      <div class="navbar__links" :class="{ open: mobileOpen }">
        <a
          v-for="link in navLinks"
          :key="link.href"
          :href="link.href"
          class="navbar__link"
          @click.prevent="scrollTo(link.href)"
        >
          {{ link.label }}
        </a>
      </div>

      <!-- Actions -->
      <div class="navbar__actions">
        <!-- Theme Toggle -->
        <button
          class="navbar__theme-btn"
          :aria-label="`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`"
          @click="toggleTheme"
        >
          <!-- Sun icon -->
          <svg v-if="theme === 'dark'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
          <!-- Moon icon -->
          <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>

        <!-- Mobile Hamburger -->
        <button class="navbar__hamburger" :class="{ active: mobileOpen }" @click="mobileOpen = !mobileOpen" aria-label="Toggle menu">
          <span></span><span></span><span></span>
        </button>
      </div>
    </div>

    <!-- Mobile Overlay -->
    <Transition name="mobile-menu">
      <div v-if="mobileOpen" class="navbar__mobile-overlay" @click="mobileOpen = false">
        <div class="navbar__mobile-menu" @click.stop>
          <a
            v-for="link in navLinks"
            :key="link.href"
            :href="link.href"
            class="navbar__mobile-link"
            @click.prevent="scrollTo(link.href)"
          >
            {{ link.label }}
          </a>
        </div>
      </div>
    </Transition>
  </nav>
</template>

<style scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: var(--z-nav);
  padding: 1rem 0;
  transition: all var(--dur-normal) var(--ease);
}

.navbar--scrolled {
  padding: 0.6rem 0;
  background: color-mix(in srgb, var(--bg-base) 80%, transparent);
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  border-bottom: 1px solid var(--border-1);
}

.navbar__inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
}

/* Logo */
.navbar__logo {
  display: flex;
  align-items: center;
  font-weight: 800;
  font-size: var(--fs-xl);
  letter-spacing: var(--tracking-wider);
  color: var(--text-1);
  text-decoration: none;
}

.navbar__logo-v {
  background: var(--gradient-brand);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 1.3em;
}

.navbar__logo-text {
  margin-left: 0.02em;
}

/* Desktop Links */
.navbar__links {
  display: flex;
  gap: 0.5rem;
}

.navbar__link {
  padding: 0.4em 0.9em;
  font-size: var(--fs-sm);
  font-weight: 500;
  color: var(--text-2);
  text-decoration: none;
  border-radius: var(--radius-sm);
  transition: all var(--dur-fast) var(--ease);
}

.navbar__link:hover {
  color: var(--text-1);
  background: var(--bg-card);
}

/* Actions */
.navbar__actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.navbar__theme-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-sm);
  color: var(--text-2);
  transition: all var(--dur-fast) var(--ease);
}

.navbar__theme-btn:hover {
  color: var(--accent);
  background: var(--bg-card);
}

/* Hamburger */
.navbar__hamburger {
  display: none;
  flex-direction: column;
  gap: 4px;
  width: 36px;
  height: 36px;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-sm);
}

.navbar__hamburger span {
  display: block;
  width: 18px;
  height: 2px;
  background: var(--text-2);
  border-radius: 2px;
  transition: all var(--dur-normal) var(--ease);
}

.navbar__hamburger.active span:nth-child(1) {
  transform: translateY(6px) rotate(45deg);
}
.navbar__hamburger.active span:nth-child(2) {
  opacity: 0;
}
.navbar__hamburger.active span:nth-child(3) {
  transform: translateY(-6px) rotate(-45deg);
}

/* Mobile */
.navbar__mobile-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(4px);
  z-index: var(--z-overlay);
}

.navbar__mobile-menu {
  position: absolute;
  top: 70px;
  left: var(--container-px);
  right: var(--container-px);
  background: var(--bg-raised);
  border: 1px solid var(--border-1);
  border-radius: var(--radius-lg);
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.navbar__mobile-link {
  padding: 0.8em 1em;
  font-size: var(--fs-base);
  font-weight: 500;
  color: var(--text-2);
  text-decoration: none;
  border-radius: var(--radius-sm);
  transition: all var(--dur-fast) var(--ease);
}

.navbar__mobile-link:hover {
  color: var(--text-1);
  background: var(--bg-card);
}

/* Mobile transition */
.mobile-menu-enter-active,
.mobile-menu-leave-active {
  transition: opacity var(--dur-normal) var(--ease);
}
.mobile-menu-enter-active .navbar__mobile-menu,
.mobile-menu-leave-active .navbar__mobile-menu {
  transition: transform var(--dur-normal) var(--ease), opacity var(--dur-normal) var(--ease);
}
.mobile-menu-enter-from,
.mobile-menu-leave-to {
  opacity: 0;
}
.mobile-menu-enter-from .navbar__mobile-menu,
.mobile-menu-leave-to .navbar__mobile-menu {
  opacity: 0;
  transform: translateY(-10px) scale(0.97);
}

@media (max-width: 768px) {
  .navbar__links { display: none; }
  .navbar__hamburger { display: flex; }
}
</style>
