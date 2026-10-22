/**
 * sw.js  —  Gepato-AI v2.0 Service Worker
 *
 * Strategiyalar:
 *   1. STATIC SHELL  — App HTML/CSS/JS: Cache-First (versiya bilan)
 *   2. API CALLS     — Network-First, failed → IndexedDB queue
 *   3. AUDIO UPLOAD  — Background Sync queue (offline yozilsa → sync)
 *   4. PUSH NOTIFY   — HE alert push xabarnomalar
 *
 * Offline flow:
 *   Bemor offline ovoz yozsa → IndexedDB ga saqlaydi
 *   → Online bo'lganda BackgroundSync ishga tushadi
 *   → /api/voice/upload ga avtomatik yuklaydi
 */

'use strict';

const CACHE_VERSION = 'gepato-v2.0.1';
const STATIC_CACHE  = CACHE_VERSION + '-static';
const DATA_CACHE    = CACHE_VERSION + '-data';

const SYNC_TAG_VOICE     = 'gepato-voice-upload';
const SYNC_TAG_COGNITIVE = 'gepato-cognitive-submit';
const SYNC_TAG_TREMOR    = 'gepato-tremor-submit';

// ── Static assets to precache ─────────────────────────────
const PRECACHE_URLS = [
  '/',
  '/patient',
  '/static/patient_daily_flow.html',
  'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap',
];

// ── API routes: network-first ─────────────────────────────
const API_PREFIXES = [
  '/api/voice/',
  '/api/tremor/',
  '/api/cognitive/',
  '/api/trend/',
  '/api/patients/',
];

// ── IndexedDB helpers ─────────────────────────────────────
const DB_NAME    = 'gepato-offline';
const DB_VERSION = 1;
const STORES = ['voice_queue', 'cognitive_queue', 'tremor_queue', 'pending_alerts'];

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => {
      const db = e.target.result;
      STORES.forEach(name => {
        if (!db.objectStoreNames.contains(name)) {
          db.createObjectStore(name, { keyPath: 'id', autoIncrement: true });
        }
      });
    };
    req.onsuccess = e => resolve(e.target.result);
    req.onerror   = e => reject(e.target.error);
  });
}

async function dbPut(store, record) {
  const db  = await openDB();
  const tx  = db.transaction(store, 'readwrite');
  const st  = tx.objectStore(store);
  return new Promise((res, rej) => {
    const req = st.put(record);
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

async function dbGetAll(store) {
  const db  = await openDB();
  const tx  = tx = db.transaction(store, 'readonly');
  const st  = tx.objectStore(store);
  return new Promise((res, rej) => {
    const req = st.getAll();
    req.onsuccess = () => res(req.result || []);
    req.onerror   = () => rej(req.error);
  });
}

async function dbDelete(store, id) {
  const db = await openDB();
  const tx = db.transaction(store, 'readwrite');
  const st = tx.objectStore(store);
  return new Promise((res, rej) => {
    const req = st.delete(id);
    req.onsuccess = () => res();
    req.onerror   = () => rej(req.error);
  });
}

// ═════════════════════════════════════════════════════════
// INSTALL — precache static shell
// ═════════════════════════════════════════════════════════
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .catch(err => console.warn('[SW] Precache error:', err))
  );
});

// ═════════════════════════════════════════════════════════
// ACTIVATE — clean old caches
// ═════════════════════════════════════════════════════════
self.addEventListener('activate', event => {
  event.waitUntil(
    Promise.all([
      clients.claim(),
      caches.keys().then(keys =>
        Promise.all(
          keys
            .filter(k => k.startsWith('gepato-') && k !== STATIC_CACHE && k !== DATA_CACHE)
            .map(k => caches.delete(k))
        )
      ),
    ])
  );
});

// ═════════════════════════════════════════════════════════
// FETCH — routing strategy
// ═════════════════════════════════════════════════════════
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET or cross-origin (except fonts)
  if (request.method !== 'GET' && !_isUpload(request)) return;
  if (url.origin !== self.location.origin && !url.hostname.includes('fonts.')) return;

  // API routes: network-first
  if (API_PREFIXES.some(p => url.pathname.startsWith(p))) {
    event.respondWith(_networkFirst(request));
    return;
  }

  // Static assets: cache-first
  event.respondWith(_cacheFirst(request));
});

function _isUpload(req) {
  return req.method === 'POST' && API_PREFIXES.some(p => new URL(req.url).pathname.startsWith(p));
}

async function _cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline — kesh topilmadi', { status: 503 });
  }
}

async function _networkFirst(request) {
  try {
    const response = await fetch(request, { signal: AbortSignal.timeout(8000) });
    if (response.ok) {
      const cache = await caches.open(DATA_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(
      JSON.stringify({ error: 'offline', message: 'Tarmoq aloqasi yo\'q' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

// ═════════════════════════════════════════════════════════
// BACKGROUND SYNC — flush offline queue
// ═════════════════════════════════════════════════════════
self.addEventListener('sync', event => {
  if (event.tag === SYNC_TAG_VOICE) {
    event.waitUntil(_flushVoiceQueue());
  } else if (event.tag === SYNC_TAG_COGNITIVE) {
    event.waitUntil(_flushCognitiveQueue());
  } else if (event.tag === SYNC_TAG_TREMOR) {
    event.waitUntil(_flushTremorQueue());
  }
});

async function _flushVoiceQueue() {
  const items = await dbGetAll('voice_queue');
  for (const item of items) {
    try {
      // Reconstruct FormData from stored base64 audio blobs
      const formData = new FormData();
      formData.append('patient_id', item.patient_id);
      formData.append('disease_stage', item.disease_stage);
      formData.append('audio_type', item.audio_type);
      if (item.audio_blob_b64) {
        const bytes = _b64ToBytes(item.audio_blob_b64);
        formData.append('audio_file', new Blob([bytes], { type:'audio/webm' }), 'recording.webm');
      }
      const res = await fetch('/api/voice/upload', { method:'POST', body:formData });
      if (res.ok) {
        await dbDelete('voice_queue', item.id);
        _notifyClients({ type:'sync_ok', store:'voice', id:item.id });
      }
    } catch (err) {
      console.warn('[SW] Voice sync failed for item', item.id, err);
    }
  }
}

async function _flushCognitiveQueue() {
  const items = await dbGetAll('cognitive_queue');
  for (const item of items) {
    try {
      const res = await fetch('/api/cognitive/nct/submit', {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify(item.payload),
      });
      if (res.ok) {
        await dbDelete('cognitive_queue', item.id);
        // Also submit stroop if present
        if (item.stroop_payload) {
          await fetch('/api/cognitive/stroop/submit', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(item.stroop_payload),
          });
        }
        _notifyClients({ type:'sync_ok', store:'cognitive', id:item.id });
      }
    } catch (err) {
      console.warn('[SW] Cognitive sync failed', item.id, err);
    }
  }
}

async function _flushTremorQueue() {
  const items = await dbGetAll('tremor_queue');
  for (const item of items) {
    try {
      const res = await fetch('/api/tremor/analyze', {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify(item.payload),
      });
      if (res.ok) {
        await dbDelete('tremor_queue', item.id);
        _notifyClients({ type:'sync_ok', store:'tremor', id:item.id });
      }
    } catch (err) {
      console.warn('[SW] Tremor sync failed', item.id, err);
    }
  }
}

// ═════════════════════════════════════════════════════════
// PUSH NOTIFICATIONS — HE alerts from server
// ═════════════════════════════════════════════════════════
self.addEventListener('push', event => {
  if (!event.data) return;
  let data;
  try { data = event.data.json(); }
  catch { data = { title:'Gepato-AI', body: event.data.text() }; }

  const ICONS = {
    critical: '🚨',
    alert:    '🔶',
    warning:  '⚠️',
    info:     'ℹ️',
  };

  const icon  = ICONS[data.level] || 'ℹ️';
  const title = data.title || 'Gepato-AI Ogohlantirish';
  const body  = data.body  || 'Yangi natija mavjud';

  const options = {
    body,
    icon:  '/static/icons/icon-192.png',
    badge: '/static/icons/badge-72.png',
    tag:   data.level === 'critical' ? 'he-critical' : 'he-alert',
    requireInteraction: data.level === 'critical',
    vibrate: data.level === 'critical' ? [200,100,200,100,400] : [100,50,100],
    data: {
      url:       data.url || '/patient',
      patient_id: data.patient_id,
      alert_level: data.level,
    },
    actions: [
      { action:'view',    title:'Ko\'rish' },
      { action:'dismiss', title:'Yopish'   },
    ],
  };

  event.waitUntil(
    self.registration.showNotification(`${icon} ${title}`, options)
  );
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  if (event.action === 'dismiss') return;

  const url = (event.notification.data || {}).url || '/patient';
  event.waitUntil(
    clients.matchAll({ type:'window', includeUncontrolled:true })
      .then(windowClients => {
        const existing = windowClients.find(c => c.url.includes('/patient'));
        if (existing) { existing.focus(); return; }
        return clients.openWindow(url);
      })
  );
});

// ═════════════════════════════════════════════════════════
// MESSAGE — commands from main thread
// ═════════════════════════════════════════════════════════
self.addEventListener('message', async event => {
  const { type, payload } = event.data || {};

  switch (type) {
    // Store voice blob offline for later sync
    case 'QUEUE_VOICE_UPLOAD': {
      await dbPut('voice_queue', {
        patient_id:    payload.patient_id,
        disease_stage: payload.disease_stage,
        audio_type:    payload.audio_type,
        audio_blob_b64: payload.audio_blob_b64,
        queued_at:     new Date().toISOString(),
      });
      // Register background sync
      if (self.registration.sync) {
        await self.registration.sync.register(SYNC_TAG_VOICE);
      }
      event.source?.postMessage({ type:'QUEUED_OK', store:'voice' });
      break;
    }

    case 'QUEUE_COGNITIVE_SUBMIT': {
      await dbPut('cognitive_queue', {
        payload:        payload.nct_payload,
        stroop_payload: payload.stroop_payload,
        queued_at:      new Date().toISOString(),
      });
      if (self.registration.sync) {
        await self.registration.sync.register(SYNC_TAG_COGNITIVE);
      }
      event.source?.postMessage({ type:'QUEUED_OK', store:'cognitive' });
      break;
    }

    case 'QUEUE_TREMOR_SUBMIT': {
      await dbPut('tremor_queue', {
        payload:   payload,
        queued_at: new Date().toISOString(),
      });
      if (self.registration.sync) {
        await self.registration.sync.register(SYNC_TAG_TREMOR);
      }
      event.source?.postMessage({ type:'QUEUED_OK', store:'tremor' });
      break;
    }

    case 'GET_QUEUE_STATUS': {
      const [voice, cog, tremor] = await Promise.all([
        dbGetAll('voice_queue'),
        dbGetAll('cognitive_queue'),
        dbGetAll('tremor_queue'),
      ]);
      event.source?.postMessage({
        type: 'QUEUE_STATUS',
        counts: { voice: voice.length, cognitive: cog.length, tremor: tremor.length },
      });
      break;
    }

    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
  }
});

// ═════════════════════════════════════════════════════════
// HELPERS
// ═════════════════════════════════════════════════════════
function _notifyClients(msg) {
  clients.matchAll({ includeUncontrolled: true })
    .then(all => all.forEach(c => c.postMessage(msg)));
}

function _b64ToBytes(b64) {
  const bin   = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
