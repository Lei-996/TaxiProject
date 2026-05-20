/* global deck, maplibregl */
(function () {
    const DEFAULT_CENTER = [116.51, 39.92];
    const DEFAULT_ZOOM = 11;

    const MAP_STYLES = [
        'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        'https://demotiles.maplibre.org/style.json',
    ];

    let map = null;
    let deckOverlay = null;
    let f2DynamicLod = true;
    let lockViewport = false;
    let debounceTimer = null;
    let abortController = null;
    let lastViewportKey = null;
    let viewportFetchGen = 0;
    let viewportLoading = false;
    let tooltipText = '车辆ID: {taxi_id}';
    let styleIndex = 0;
    const DEBOUNCE_MS = 280;

    function buildLayers(layerSpecs) {
        const layers = [];
        for (const spec of layerSpecs || []) {
            if (spec.type === 'scatter') {
                layers.push(new deck.ScatterplotLayer({
                    id: spec.id,
                    data: spec.data,
                    pickable: true,
                    opacity: spec.opacity ?? 0.65,
                    stroked: false,
                    filled: true,
                    radiusScale: 1,
                    radiusMinPixels: 2,
                    radiusMaxPixels: 48,
                    pickable: spec.pickable !== false,
                    getPosition: (d) => [d.lon, d.lat],
                    getRadius: spec.radius ?? 28,
                    getFillColor: (d) => d.color || spec.color || [65, 150, 210, 180],
                }));
            } else if (spec.type === 'path') {
                layers.push(new deck.PathLayer({
                    id: spec.id,
                    data: spec.data,
                    pickable: true,
                    widthScale: 12,
                    widthMinPixels: 2,
                    getPath: (d) => d.path,
                    getColor: (d) => d.color || [255, 100, 50, 220],
                    getWidth: (d) => d.width || 3,
                    opacity: 0.88,
                }));
            } else if (spec.type === 'column') {
                layers.push(new deck.ColumnLayer({
                    id: spec.id,
                    data: spec.data,
                    diskResolution: 8,
                    radius: 380,
                    extruded: true,
                    pickable: true,
                    elevationScale: spec.elevation_scale ?? 1,
                    getPosition: (d) => [d.lon, d.lat],
                    getFillColor: (d) => d.color,
                    getElevation: (d) => d.count,
                }));
            } else if (spec.type === 'heatmap') {
                layers.push(new deck.HeatmapLayer({
                    id: spec.id,
                    data: spec.data,
                    pickable: true,
                    radiusPixels: spec.radius_pixels ?? 50,
                    intensity: spec.intensity ?? 1,
                    threshold: spec.threshold ?? 0.05,
                    colorRange: [
                        [33, 102, 172, 0],
                        [67, 147, 195, 180],
                        [146, 197, 222, 200],
                        [209, 229, 240, 220],
                        [253, 219, 199, 230],
                        [246, 178, 107, 240],
                        [239, 138, 98, 250],
                        [215, 48, 39, 255],
                    ],
                    getPosition: (d) => [d.lon, d.lat],
                    getWeight: (d) => d.weight ?? d.count ?? 1,
                }));
            } else if (spec.type === 'arc') {
                layers.push(new deck.ArcLayer({
                    id: spec.id,
                    data: spec.data,
                    pickable: true,
                    greatCircle: true,
                    numSegments: 64,
                    getWidth: (d) => d.width || 4,
                    getSourcePosition: (d) => d.source,
                    getTargetPosition: (d) => d.target,
                    getSourceColor: (d) => d.color,
                    getTargetColor: (d) => d.color,
                    getHeight: (d) => d.height ?? 0.4,
                }));
            } else if (spec.type === 'polygon') {
                layers.push(new deck.PolygonLayer({
                    id: spec.id,
                    data: spec.data,
                    pickable: false,
                    stroked: true,
                    filled: true,
                    wireframe: false,
                    lineWidthMinPixels: 2,
                    getPolygon: (d) => d.polygon,
                    getFillColor: (d) => d.fill_color,
                    getLineColor: (d) => d.line_color,
                }));
            }
        }
        return layers;
    }

    function formatTooltip(info) {
        if (!info.object) return '';
        const o = info.object;
        return tooltipText
            .replace('{taxi_id}', o.taxi_id ?? '')
            .replace('{type}', o.type ?? '')
            .replace('{label}', o.label ?? '')
            .replace('{count}', o.count ?? '');
    }

    function getCurrentView() {
        if (!map) {
            return { longitude: DEFAULT_CENTER[0], latitude: DEFAULT_CENTER[1], zoom: DEFAULT_ZOOM };
        }
        const c = map.getCenter();
        return { longitude: c.lng, latitude: c.lat, zoom: map.getZoom() };
    }

    function updateF2Status(meta) {
        const el = document.getElementById('f2_status');
        if (!el) return;
        const view = getCurrentView();
        const z = meta?.zoom != null ? meta.zoom : view.zoom;
        const n = meta?.point_count != null ? meta.point_count.toLocaleString() : '-';
        const s = meta?.sample_stride != null ? meta.sample_stride : '-';
        el.textContent = `zoom ${Number(z).toFixed(1)} | 点数 ${n} | 采样步长 ${s}`;
    }

    function getMapBbox() {
        if (!map) return null;
        const b = map.getBounds();
        return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    }

    function viewportKey(view) {
        const bbox = getMapBbox();
        if (bbox) {
            return `${view.zoom.toFixed(2)}|${bbox.map((v) => v.toFixed(4)).join(',')}`;
        }
        return `${view.zoom.toFixed(2)}|${view.longitude.toFixed(4)}|${view.latitude.toFixed(4)}`;
    }

    async function fetchViewport(view) {
        const container = document.getElementById('map');
        const bbox = getMapBbox();
        if (abortController) abortController.abort();
        abortController = new AbortController();
        const body = {
            zoom: view.zoom,
            longitude: view.longitude,
            latitude: view.latitude,
        };
        if (bbox) {
            body.bbox = bbox;
        } else {
            body.width = Math.max(container?.clientWidth || 0, 400);
            body.height = Math.max(container?.clientHeight || 0, 300);
        }
        const resp = await fetch('/api/map_viewport', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        return data;
    }

    function scheduleViewportFetch() {
        if (!f2DynamicLod || lockViewport || !map) return;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const view = getCurrentView();
            const key = viewportKey(view);
            if (key === lastViewportKey) return;
            const gen = ++viewportFetchGen;
            viewportLoading = true;
            try {
                const data = await fetchViewport(view);
                if (gen !== viewportFetchGen) return;
                lastViewportKey = key;
                applyMapPayload(data.map, false);
                updateF2Status(data.map.meta);
            } catch (e) {
                if (e.name !== 'AbortError') {
                    console.warn('F2 viewport load failed', e);
                }
            } finally {
                if (gen === viewportFetchGen) viewportLoading = false;
            }
        }, DEBOUNCE_MS);
    }

    function jumpToViewState(vs) {
        if (!map || !vs) return;
        map.jumpTo({
            center: [vs.longitude, vs.latitude],
            zoom: vs.zoom ?? DEFAULT_ZOOM,
            pitch: vs.pitch ?? 0,
            bearing: vs.bearing ?? 0,
        });
    }

    function applyMapPayload(payload, updateViewFromPayload) {
        if (!payload || !deckOverlay) return;
        lockViewport = !!payload.lockViewport;
        if (lockViewport) {
            lastViewportKey = viewportKey(getCurrentView());
        }
        tooltipText = payload.tooltip || tooltipText;
        const layers = buildLayers(payload.layers);
        deckOverlay.setProps({ layers });

        if (updateViewFromPayload && payload.viewState) {
            jumpToViewState(payload.viewState);
        }
        if (payload.meta) updateF2Status(payload.meta);
        map?.resize();
    }

    function tryFallbackStyle() {
        if (!map || styleIndex >= MAP_STYLES.length - 1) return;
        styleIndex += 1;
        map.setStyle(MAP_STYLES[styleIndex]);
    }

    function initMap(initialPayload) {
        const container = document.getElementById('map');
        if (!container || typeof maplibregl === 'undefined' || typeof deck === 'undefined') {
            console.error('maplibre-gl 或 deck.gl 未加载');
            return;
        }

        const cb = document.getElementById('f2_dynamic_lod');
        if (cb) {
            f2DynamicLod = cb.checked;
            cb.addEventListener('change', () => {
                f2DynamicLod = cb.checked;
                if (f2DynamicLod && !lockViewport) scheduleViewportFetch();
            });
        }

        map = new maplibregl.Map({
            container,
            style: MAP_STYLES[styleIndex],
            center: DEFAULT_CENTER,
            zoom: DEFAULT_ZOOM,
            pitch: 0,
            bearing: 0,
            attributionControl: true,
        });

        map.on('error', (e) => {
            if (e?.error?.message) console.warn('地图样式加载失败，尝试备用源', e.error.message);
            tryFallbackStyle();
        });

        map.on('load', () => {
            deckOverlay = new deck.MapboxOverlay({
                interleaved: true,
                layers: [],
                getTooltip: ({ object }) => (object ? formatTooltip({ object }) : null),
            });
            map.addControl(deckOverlay);

            map.on('movestart', () => clearTimeout(debounceTimer));
            map.on('moveend', scheduleViewportFetch);
            map.on('zoomend', scheduleViewportFetch);
            window.addEventListener('resize', () => {
                map.resize();
                lastViewportKey = null;
                scheduleViewportFetch();
            });

            if (initialPayload) {
                applyMapPayload(initialPayload, true);
            } else {
                scheduleViewportFetch();
            }
        });
    }

    window.applyMapFromApi = function (data) {
        if (data.map) {
            applyMapPayload(data.map, true);
        }
    };

    window.resetScatterViewport = async function () {
        lockViewport = false;
        lastViewportKey = null;
        if (typeof updateResult === 'function') {
            updateResult('⏳ 恢复散点视图...', 'info');
        }
        try {
            const resp = await fetch('/api/normal_view');
            const data = await resp.json();
            applyMapPayload(data.map, true);
            if (typeof updateResult === 'function') {
                updateResult('✅ 已恢复散点视图（F2 动态加载已启用）', 'success');
            }
        } catch (e) {
            if (typeof updateResult === 'function') {
                updateResult('❌ 恢复失败', 'error');
            }
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        initMap(window.INITIAL_MAP || null);
    });
})();
