/* Mira Showcase — Lightweight Persistent Asset Cache (Service Worker)
 * - Images & static assets (/assets/*, *.webp, *.svg): Cache-First (beats GitHub Pages' 10-min TTL)
 * - HTML / CSS / JS: Network-First with offline cache fallback
 * - Versioned cache name automatically purges older entries on deploy. */
"use strict";

const CACHE_NAME = "mira-assets-v93";
const HERO_PRECACHE = ["./assets/bg/mira-underwater-scene.webp"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(HERO_PRECACHE).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith("mira-assets-") && k !== CACHE_NAME)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const isAsset =
    url.pathname.includes("/assets/") ||
    /\.(webp|png|jpe?g|svg|woff2?)$/i.test(url.pathname);

  if (isAsset) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) =>
        cache.match(req).then((hit) => {
          if (hit) return hit;
          return fetch(req).then((res) => {
            if (res && res.ok && res.type === "basic") {
              cache.put(req, res.clone());
            }
            return res;
          });
        })
      )
    );
    return;
  }

  const isAppShell =
    url.pathname.endsWith("/") || /\.(html|css|js|json)$/i.test(url.pathname);
  if (!isAppShell) return;

  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok && res.type === "basic") {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req))
  );
});
