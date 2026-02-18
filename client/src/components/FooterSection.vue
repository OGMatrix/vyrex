<script setup lang="ts">
/**
 * FooterSection — Site Footer with Links & Credits
 */
import type { RunInfo } from '../types'

defineProps<{
  runInfo: RunInfo | null
}>()
</script>

<template>
  <footer class="footer">
    <div class="footer__divider" aria-hidden="true"></div>
    <div class="container">
      <div class="footer__inner">
        <div class="footer__brand">
          <span class="footer__logo">
            <span class="footer__logo-v">V</span>YREX
          </span>
          <p class="footer__tagline">
            Devastating 2v2 Rocket League AI — trained with PPO via RLGym v2 + RocketSim
          </p>
        </div>

        <div class="footer__links">
          <div class="footer__col">
            <h4 class="footer__col-title">Project</h4>
            <a href="#vision" class="footer__link">Vision</a>
            <a href="#metrics" class="footer__link">Live Metrics</a>
            <a href="#journey" class="footer__link">Journey</a>
            <a href="#stack" class="footer__link">Tech Stack</a>
          </div>
          <div class="footer__col">
            <h4 class="footer__col-title">Resources</h4>
            <a href="https://rlgym.org" target="_blank" rel="noopener" class="footer__link">RLGym ↗</a>
            <a href="https://wandb.ai" target="_blank" rel="noopener" class="footer__link">Weights & Biases ↗</a>
            <a href="https://rlbot.org" target="_blank" rel="noopener" class="footer__link">RLBot ↗</a>
          </div>
        </div>
      </div>

      <div class="footer__bottom">
        <span class="footer__run-info font-mono" v-if="runInfo">
          Run {{ runInfo.id }} · {{ runInfo.state }}
        </span>
        <span class="footer__copy">
          © {{ new Date().getFullYear() }} VYREX · Built with dedication
        </span>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.footer {
  position: relative;
  background: var(--bg-overlay);
  padding-top: 5rem;
  padding-bottom: 2rem;
}

.footer__divider {
  position: absolute;
  top: -1px;
  left: 0;
  right: 0;
  height: 60px;
  background: var(--bg-base);
  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 40%);
}

.footer__inner {
  display: flex;
  justify-content: space-between;
  gap: 3rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--border-1);
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}

.footer__brand {
  max-width: 350px;
}

.footer__logo {
  font-weight: 800;
  font-size: var(--fs-xl);
  letter-spacing: var(--tracking-wider);
}

.footer__logo-v {
  background: var(--gradient-brand);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.footer__tagline {
  margin-top: 0.75rem;
  font-size: var(--fs-sm);
  color: var(--text-3);
  line-height: var(--leading-normal);
}

.footer__links {
  display: flex;
  gap: 3rem;
}

.footer__col {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.footer__col-title {
  font-size: var(--fs-xs);
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wider);
  margin-bottom: 0.25rem;
}

.footer__link {
  font-size: var(--fs-sm);
  color: var(--text-2);
  text-decoration: none;
  transition: color var(--dur-fast) var(--ease);
}

.footer__link:hover {
  color: var(--accent);
}

.footer__bottom {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.footer__run-info {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

.footer__copy {
  font-size: var(--fs-xs);
  color: var(--text-3);
}

@media (max-width: 640px) {
  .footer__links {
    gap: 2rem;
  }
}
</style>
