/**
 * pwa_register.js  —  Gepato-AI v2.0
 * Barcha patient sahifalariga <script src="/static/pwa_register.js"> bilan qo'shiladi.
 *
 * Qiladi:
 *   1. SW ro'yxatdan o'tkazadi
 *   2. Offline queue statusini ko'rsatadi
 *   3. Sync update banneri chiqaradi
 *   4. Push notification ruxsati so'raydi (faqat bir marta)
 */

(function () {
  'use strict';

  // ── SW Registration ──────────────────────────────────────
  if (!('serviceWorker' in navigator)) {
    console.info('[PWA] Service Worker not supported');
    return;
  }

  let swReg = null;

  navigator.serviceWorker
    .register('/static/sw.js', { scope: '/' })
    .then(reg => {
      swReg = reg;
      console.info('[PWA] SW registered, scope:', reg.scope);
      _listenForUpdates(reg);
      _checkQueueStatus();
    })
    .catch(err => console.warn('[PWA] SW registration failed:', err));

  // ── Listen for SW updates ────────────────────────────────
  function _listenForUpdates(reg) {
    reg.addEventListener('updatefound', () => {
      const newWorker = reg.installing;
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          _showUpdateBanner();
        }
      });
    });
  }

  function _showUpdateBanner() {
    const banner = document.createElement('div');
    banner.id = 'pwa-update-banner';
    banner.style.cssText = [
      'position:fixed', 'bottom:16px', 'left:50%', 'transform:translateX(-50%)',
      'background:#030213', 'color:#fff', 'padding:10px 16px',
      'border-radius:100px', 'font-size:13px',
      'display:flex', 'align-items:center', 'gap:10px',
      'box-shadow:0 4px 12px rgba(0,0,0,0.2)', 'z-index:9999',
      'font-family:Geist,sans-serif',
    ].join(';');
    banner.innerHTML = `
      <span>Yangi versiya mavjud</span>
      <button onclick="window.PWA.applyUpdate()" style="
        background:#1A56DB;color:#fff;border:none;padding:5px 12px;
        border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;
        font-family:Geist,sans-serif;
      ">Yangilash</button>
    `;
    document.body.appendChild(banner);
  }

  // ── Offline queue status indicator ──────────────────────
  function _checkQueueStatus() {
    if (!navigator.serviceWorker.controller) return;
    navigator.serviceWorker.controller.postMessage({ type: 'GET_QUEUE_STATUS' });
  }

  navigator.serviceWorker.addEventListener('message', event => {
    const { type, counts, store } = event.data || {};

    if (type === 'QUEUE_STATUS') {
      const total = (counts.voice || 0) + (counts.cognitive || 0) + (counts.tremor || 0);
      if (total > 0) _showOfflineBadge(total);
    }

    if (type === 'sync_ok') {
      _removeOfflineBadge();
      _checkQueueStatus();
    }

    if (type === 'QUEUED_OK') {
      console.info(`[PWA] Queued for background sync: ${store}`);
    }
  });

  // ── Offline status badge ─────────────────────────────────
  function _showOfflineBadge(count) {
    let badge = document.getElementById('pwa-offline-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'pwa-offline-badge';
      badge.style.cssText = [
        'position:fixed', 'top:12px', 'right:12px',
        'background:#F59E0B', 'color:#fff',
        'padding:4px 10px', 'border-radius:100px',
        'font-size:12px', 'font-weight:500', 'z-index:9999',
        'font-family:Geist,sans-serif',
        'display:flex', 'align-items:center', 'gap:5px',
      ].join(';');
      document.body.appendChild(badge);
    }
    badge.innerHTML = `
      <span>⏳</span>
      <span>${count} ta yuborilmagan (offline)</span>
    `;
  }

  function _removeOfflineBadge() {
    const badge = document.getElementById('pwa-offline-badge');
    if (badge) badge.remove();
  }

  // ── Network status listener ──────────────────────────────
  window.addEventListener('online',  () => {
    _removeOfflineBadge();
    // Trigger manual sync attempt
    if (swReg && swReg.sync) {
      ['gepato-voice-upload','gepato-cognitive-submit','gepato-tremor-submit']
        .forEach(tag => swReg.sync.register(tag).catch(()=>{}));
    }
    // Re-check queue
    setTimeout(_checkQueueStatus, 1500);
  });

  window.addEventListener('offline', () => {
    _checkQueueStatus();
  });

  // ── Push notification permission ─────────────────────────
  // Faqat critical alert darajasidagi bemorlarda so'raladi
  // window.PWA.requestPush() ni shifokor tizimidan chaqirish mumkin
  async function _requestPushPermission() {
    if (!('Notification' in window) || !('PushManager' in window)) return null;
    if (Notification.permission === 'granted') return _subscribePush();
    if (Notification.permission === 'denied')  return null;
    const perm = await Notification.requestPermission();
    if (perm === 'granted') return _subscribePush();
    return null;
  }

  async function _subscribePush() {
    if (!swReg) return null;
    try {
      // VAPID public key: real app → fetch from /api/push/vapid-public-key
      const VAPID_PUBLIC_KEY = window.VAPID_PUBLIC_KEY || 'REPLACE_WITH_REAL_VAPID_KEY';
      const sub = await swReg.pushManager.subscribe({
        userVisibleOnly:      true,
        applicationServerKey: _urlBase64ToUint8Array(VAPID_PUBLIC_KEY),
      });
      // POST subscription to backend
      await fetch('/api/push/subscribe', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          subscription: sub.toJSON(),
          patient_id:   window.PATIENT_ID || null,
        }),
      });
      console.info('[PWA] Push subscribed');
      return sub;
    } catch (err) {
      console.warn('[PWA] Push subscribe failed:', err);
      return null;
    }
  }

  function _urlBase64ToUint8Array(base64String) {
    const padding   = '='.repeat((4 - base64String.length % 4) % 4);
    const base64    = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData   = atob(base64);
    const outputArr = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; i++) outputArr[i] = rawData.charCodeAt(i);
    return outputArr;
  }

  // ── Offline voice upload helper ──────────────────────────
  // Patient daily flow sahifasida ishlatiladi:
  //   window.PWA.queueVoiceUpload(blob, patientId, stage, audioType)
  async function _queueVoiceUpload(blob, patientId, stage, audioType) {
    // Convert Blob to base64 (for IndexedDB storage via SW)
    const b64 = await _blobToBase64(blob);
    navigator.serviceWorker.controller?.postMessage({
      type: 'QUEUE_VOICE_UPLOAD',
      payload: {
        patient_id:    patientId,
        disease_stage: stage,
        audio_type:    audioType,
        audio_blob_b64: b64,
      },
    });
  }

  function _blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
      reader.onerror   = reject;
      reader.readAsDataURL(blob);
    });
  }

  // ── Public API ───────────────────────────────────────────
  window.PWA = {
    queueVoiceUpload:      _queueVoiceUpload,
    requestPushPermission: _requestPushPermission,
    checkQueueStatus:      _checkQueueStatus,
    applyUpdate() {
      if (swReg && swReg.waiting) {
        swReg.waiting.postMessage({ type: 'SKIP_WAITING' });
        window.location.reload();
      }
    },
    isOnline:  () => navigator.onLine,
    isOffline: () => !navigator.onLine,
  };

})();
