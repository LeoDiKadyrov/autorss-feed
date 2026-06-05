/* Phase 85 Plan 02 — Dwell-time beacon.
 *
 * Privacy invariants (enforced by tests/web/test_dwell_beacon_integration.py):
 *   - session_id lives in sessionStorage ONLY (no localStorage, no cookies).
 *   - Payload contains ONLY {item_id, dwell_ms, session_id}. No userAgent,
 *     no referrer, no extra timestamps.
 *   - Max 1 beacon per (item_id, session) per page-load (dedup set).
 *   - Fires only when dwell >= 3000 ms at >= 0.5 visibility.
 */
(function () {
    "use strict";

    function getSessionId() {
        var sid = sessionStorage.getItem("dwell_session_id");
        if (!sid) {
            sid = crypto.randomUUID();
            sessionStorage.setItem("dwell_session_id", sid);
        }
        return sid;
    }

    var DWELL_THRESHOLD_MS = 3000;
    var VISIBILITY_THRESHOLD = 0.5;

    var startedAt = new Map();   // itemId -> startMs (currently-visible)
    var beaconed = new Set();    // itemId -> already fired this page-load

    function sendBeacon(itemId, dwellMs) {
        if (beaconed.has(itemId)) return;
        var payload = JSON.stringify({
            item_id: itemId,
            dwell_ms: dwellMs,
            session_id: getSessionId()
        });
        var blob = new Blob([payload], { type: "application/json" });
        // WR-05 (Phase 85 review): mark `beaconed` ONLY after a confirmed
        // dispatch — otherwise a sendBeacon false return (queue full, CSP
        // block, Safari keepalive cap) silently dropped the row AND blocked
        // any retry for the rest of the page-load.
        var sent = false;
        try { sent = navigator.sendBeacon("/api/dwell", blob); } catch (e) {}
        if (sent) {
            beaconed.add(itemId);
            return;
        }
        // Fallback: fetch with keepalive so it survives unload. Only mark
        // beaconed when the server confirms 2xx; transient errors stay
        // un-beaconed so the next intersect can re-fire.
        try {
            fetch("/api/dwell", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: payload,
                keepalive: true
            }).then(function (r) {
                if (r && r.ok) beaconed.add(itemId);
                else if (typeof console !== "undefined" && console.warn) {
                    console.warn("dwell beacon fallback failed status=" + (r && r.status));
                }
            }).catch(function (e) {
                if (typeof console !== "undefined" && console.warn) {
                    console.warn("dwell beacon fallback error", e);
                }
            });
        } catch (e) {
            if (typeof console !== "undefined" && console.warn) {
                console.warn("dwell beacon dispatch error", e);
            }
        }
    }

    function onIntersect(entries) {
        for (var i = 0; i < entries.length; i++) {
            var e = entries[i];
            var raw = e.target.dataset.itemId;
            var itemId = parseInt(raw, 10);
            if (isNaN(itemId)) continue;
            if (e.isIntersecting && e.intersectionRatio >= VISIBILITY_THRESHOLD) {
                if (!startedAt.has(itemId)) {
                    startedAt.set(itemId, Date.now());
                }
            } else if (startedAt.has(itemId)) {
                var dwell = Date.now() - startedAt.get(itemId);
                startedAt.delete(itemId);
                if (dwell >= DWELL_THRESHOLD_MS) {
                    sendBeacon(itemId, dwell);
                }
            }
        }
    }

    function flushAll() {
        var now = Date.now();
        startedAt.forEach(function (startMs, itemId) {
            var dwell = now - startMs;
            if (dwell >= DWELL_THRESHOLD_MS) {
                sendBeacon(itemId, dwell);
            }
        });
        startedAt.clear();
    }

    // BL-02 (Phase 85 review): track observed elements so we can re-arm them
    // on tab return. IntersectionObserver only fires on threshold crossings —
    // a tab that was visible, hidden, then re-shown does NOT re-fire for
    // elements already in the viewport. Without explicit re-arm we silently
    // lose dwell on the most common usage pattern.
    var observed = [];

    function reArmVisible() {
        var now = Date.now();
        var vh = window.innerHeight || document.documentElement.clientHeight;
        for (var i = 0; i < observed.length; i++) {
            var el = observed[i];
            var itemId = parseInt(el.dataset.itemId, 10);
            if (isNaN(itemId)) continue;
            if (beaconed.has(itemId)) continue;
            if (startedAt.has(itemId)) continue;
            var rect = el.getBoundingClientRect();
            if (rect.height <= 0) continue;
            var visibleH = Math.max(
                0, Math.min(rect.bottom, vh) - Math.max(rect.top, 0)
            );
            if ((visibleH / rect.height) >= VISIBILITY_THRESHOLD) {
                startedAt.set(itemId, now);
            }
        }
    }

    function init() {
        if (!("IntersectionObserver" in window)) return;
        var observer = new IntersectionObserver(onIntersect, {
            threshold: [VISIBILITY_THRESHOLD]
        });
        var rows = document.querySelectorAll(".react-row[data-item-id]");
        rows.forEach(function (el) { observer.observe(el); observed.push(el); });

        document.addEventListener("visibilitychange", function () {
            if (document.visibilityState === "hidden") {
                flushAll();
            } else if (document.visibilityState === "visible") {
                reArmVisible();
            }
        });
        window.addEventListener("pagehide", flushAll);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
