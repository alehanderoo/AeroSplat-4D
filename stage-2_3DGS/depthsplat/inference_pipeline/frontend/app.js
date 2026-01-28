/**
 * DepthSplat Real-Time Visualization Frontend
 *
 * WebSocket client that connects to the inference pipeline and displays:
 * - Input camera feeds
 * - Cropped object detection
 * - Ground truth depth
 * - Monocular depth
 * - Predicted depth
 * - Silhouette/confidence
 * - 3D Gaussian Splat render
 * - Performance statistics
 */

class DepthSplatViewer {
    constructor() {
        // Configuration
        this.config = {
            wsUrl: `ws://${window.location.hostname}:8765`,
            reconnectDelay: 1000,
            maxReconnectDelay: 30000,
            numCameras: 5,
        };

        // State
        this.ws = null;
        this.connected = false;
        this.paused = false;
        this.reconnectAttempts = 0;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        this.fpsHistory = [];
        this.viewMode = 'orbit';
        this.gtDepthEnabled = false;  // GT depth disabled by default for performance

        // DOM elements cache
        this.elements = {};

        // Initialize
        this.init();
    }

    init() {
        this.cacheElements();
        this.setupInputFeeds();
        this.setupCroppedFeeds();
        this.setupGtDepthFeeds();
        this.setupMonoDepthFeeds();
        this.setupPredictedDepthFeeds();
        this.setupSilhouetteFeeds();
        this.setupEventListeners();
        this.connect();
    }

    cacheElements() {
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            fpsDisplay: document.getElementById('fps-display'),
            latencyDisplay: document.getElementById('latency-display'),
            pauseBtn: document.getElementById('pause-btn'),
            inputFeeds: document.getElementById('input-feeds'),
            croppedFeeds: document.getElementById('cropped-feeds'),
            gtDepthFeeds: document.getElementById('gt-depth-feeds'),
            gtDepthColumn: document.getElementById('gt-depth-column'),
            monoDepthFeeds: document.getElementById('mono-depth-feeds'),
            predictedDepthFeeds: document.getElementById('predicted-depth-feeds'),
            silhouetteFeeds: document.getElementById('silhouette-feeds'),
            gaussianRender: document.getElementById('gaussian-render'),
            renderPlaceholder: document.getElementById('render-placeholder'),
            viewMode: document.getElementById('view-mode'),
            toggleGtDepth: document.getElementById('toggle-gt-depth'),
            encoderTime: document.getElementById('encoder-time'),
            decoderTime: document.getElementById('decoder-time'),
            gaussianCount: document.getElementById('gaussian-count'),
            totalLatency: document.getElementById('total-latency'),
            networkLatency: document.getElementById('network-latency'),
            frameId: document.getElementById('frame-id'),
            clientCount: document.getElementById('client-count'),
            // Per-column latency displays
            latencyInput: document.getElementById('latency-input'),
            latencyCropped: document.getElementById('latency-cropped'),
            latencyGtDepth: document.getElementById('latency-gt-depth'),
            latencyMono: document.getElementById('latency-mono'),
            latencyPredicted: document.getElementById('latency-predicted'),
            latencySilhouette: document.getElementById('latency-silhouette'),
        };
    }

    setupFeedColumn(container, idPrefix, labelPrefix, containerClass = 'aspect-square') {
        container.innerHTML = '';
        for (let i = 0; i < this.config.numCameras; i++) {
            const feedDiv = document.createElement('div');
            feedDiv.className = 'relative bg-gray-800 rounded border border-gray-700 overflow-hidden';
            feedDiv.innerHTML = `
                <div class="${containerClass}">
                    <img
                        id="${idPrefix}-${i}"
                        class="stream-image w-full h-full object-cover"
                        alt="${labelPrefix} ${i + 1}"
                    />
                    <div id="${idPrefix}-placeholder-${i}" class="absolute inset-0 flex items-center justify-center bg-gray-800">
                        <span class="text-xs text-gray-500">${labelPrefix} ${i + 1}</span>
                    </div>
                </div>
            `;
            container.appendChild(feedDiv);
        }
    }

    setupInputFeeds() {
        // Input cameras use fixed height cells to match columns 2-6, with 16:9 aspect ratio
        this.setupFeedColumn(this.elements.inputFeeds, 'input-feed', 'Cam', 'input-feed-cell');
    }

    setupCroppedFeeds() {
        this.setupFeedColumn(this.elements.croppedFeeds, 'cropped-feed', 'Crop');
    }

    setupGtDepthFeeds() {
        this.setupFeedColumn(this.elements.gtDepthFeeds, 'gt-depth-feed', 'GT');
    }

    setupMonoDepthFeeds() {
        this.setupFeedColumn(this.elements.monoDepthFeeds, 'mono-depth-feed', 'Mono');
    }

    setupPredictedDepthFeeds() {
        this.setupFeedColumn(this.elements.predictedDepthFeeds, 'predicted-depth-feed', 'Pred');
    }

    setupSilhouetteFeeds() {
        this.setupFeedColumn(this.elements.silhouetteFeeds, 'silhouette-feed', 'Sil');
    }

    setupEventListeners() {
        // Pause button
        this.elements.pauseBtn.addEventListener('click', () => this.togglePause());

        // View mode selector
        this.elements.viewMode.addEventListener('change', (e) => {
            this.setViewMode(e.target.value);
        });

        // GT Depth toggle
        this.elements.toggleGtDepth.addEventListener('change', (e) => {
            this.setGtDepthEnabled(e.target.checked);
        });

        // Initialize GT depth column visibility
        this.updateGtDepthVisibility();

        // Handle visibility change (pause when hidden)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.paused) {
                // Could auto-pause when tab is hidden to save bandwidth
            }
        });

        // Handle window focus
        window.addEventListener('focus', () => {
            if (!this.connected) {
                this.connect();
            }
        });
    }

    setGtDepthEnabled(enabled) {
        this.gtDepthEnabled = enabled;
        this.updateGtDepthVisibility();

        // Send toggle state to server
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_gt_depth',
                enabled: enabled
            }));
        }
    }

    updateGtDepthVisibility() {
        if (this.elements.gtDepthColumn) {
            this.elements.gtDepthColumn.style.display = this.gtDepthEnabled ? 'flex' : 'none';
        }
        // Sync checkbox state
        if (this.elements.toggleGtDepth) {
            this.elements.toggleGtDepth.checked = this.gtDepthEnabled;
        }
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
            return;
        }

        console.log(`Connecting to ${this.config.wsUrl}...`);
        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = (event) => this.onClose(event);
            this.ws.onerror = (error) => this.onError(error);
        } catch (error) {
            console.error('WebSocket creation error:', error);
            this.scheduleReconnect();
        }
    }

    onOpen() {
        console.log('Connected to visualization server');
        this.connected = true;
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected');
    }

    onMessage(event) {
        if (this.paused) return;

        try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }

    onClose(event) {
        console.log(`Connection closed: ${event.code} ${event.reason}`);
        this.connected = false;
        this.updateConnectionStatus('disconnected');
        this.scheduleReconnect();
    }

    onError(error) {
        console.error('WebSocket error:', error);
    }

    scheduleReconnect() {
        const delay = Math.min(
            this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.config.maxReconnectDelay
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})...`);
        this.reconnectAttempts++;

        setTimeout(() => {
            if (!this.connected) {
                this.connect();
            }
        }, delay);
    }

    handleMessage(data) {
        switch (data.type) {
            case 'init':
                this.handleInit(data);
                break;
            case 'frame':
                this.handleFrame(data);
                break;
            case 'view_mode_changed':
                this.handleViewModeChanged(data);
                break;
            case 'gt_depth_changed':
                this.handleGtDepthChanged(data);
                break;
            case 'pong':
                this.handlePong(data);
                break;
            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    handleInit(data) {
        console.log('Received init:', data);

        // Update config if provided
        if (data.config) {
            if (data.config.num_cameras) {
                this.config.numCameras = data.config.num_cameras;
                this.setupInputFeeds();
                this.setupCroppedFeeds();
                this.setupGtDepthFeeds();
                this.setupMonoDepthFeeds();
                this.setupPredictedDepthFeeds();
                this.setupSilhouetteFeeds();
            }
        }

        // Sync view mode
        if (data.view_mode) {
            this.viewMode = data.view_mode;
            this.elements.viewMode.value = data.view_mode;
        }

        // Sync GT depth enabled state
        if (data.gt_depth_enabled !== undefined) {
            this.gtDepthEnabled = data.gt_depth_enabled;
            this.updateGtDepthVisibility();
        }
    }

    handleFrame(data) {
        const receiveTime = Date.now();
        const serverTime = data.timestamp;
        const networkLatency = receiveTime - serverTime;

        // Update frame counter
        this.frameCount++;
        this.elements.frameId.textContent = data.frame_id;

        // Update FPS
        this.updateFps();

        // Update input feeds
        if (data.inputs) {
            data.inputs.forEach((base64, index) => {
                this.updateImage(`input-feed-${index}`, `input-feed-placeholder-${index}`, base64);
            });
        }

        // Update cropped views
        if (data.cropped && Array.isArray(data.cropped)) {
            data.cropped.forEach((base64, index) => {
                this.updateImage(`cropped-feed-${index}`, `cropped-feed-placeholder-${index}`, base64);
            });
        }

        // Update ground truth depth
        if (data.gt_depth && Array.isArray(data.gt_depth)) {
            data.gt_depth.forEach((base64, index) => {
                this.updateImage(`gt-depth-feed-${index}`, `gt-depth-feed-placeholder-${index}`, base64);
            });
        }

        // Update monocular depth
        if (data.mono_depth && Array.isArray(data.mono_depth)) {
            data.mono_depth.forEach((base64, index) => {
                this.updateImage(`mono-depth-feed-${index}`, `mono-depth-feed-placeholder-${index}`, base64);
            });
        }

        // Update predicted depth
        if (data.predicted_depth && Array.isArray(data.predicted_depth)) {
            data.predicted_depth.forEach((base64, index) => {
                this.updateImage(`predicted-depth-feed-${index}`, `predicted-depth-feed-placeholder-${index}`, base64);
            });
        }

        // Update silhouette
        if (data.silhouette && Array.isArray(data.silhouette)) {
            data.silhouette.forEach((base64, index) => {
                this.updateImage(`silhouette-feed-${index}`, `silhouette-feed-placeholder-${index}`, base64);
            });
        }

        // Update Gaussian render
        this.updateImage('gaussian-render', 'render-placeholder', data.gaussian_render);

        // Update stats
        if (data.stats) {
            this.updateStats(data.stats, networkLatency);
        }
    }

    handleViewModeChanged(data) {
        this.viewMode = data.mode;
        this.elements.viewMode.value = data.mode;
    }

    handleGtDepthChanged(data) {
        this.gtDepthEnabled = data.enabled;
        this.updateGtDepthVisibility();
    }

    handlePong(data) {
        const rtt = Date.now() - data.timestamp;
        console.log(`Ping RTT: ${rtt}ms`);
    }

    updateImage(imgId, placeholderId, base64Data) {
        const img = document.getElementById(imgId);
        const placeholder = document.getElementById(placeholderId);

        if (!img) return;

        if (base64Data) {
            img.src = `data:image/jpeg;base64,${base64Data}`;
            img.style.display = 'block';
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        } else {
            img.style.display = 'none';
            if (placeholder) {
                placeholder.style.display = 'flex';
            }
        }
    }

    updateFps() {
        const now = Date.now();
        if (this.lastFrameTime > 0) {
            const delta = now - this.lastFrameTime;
            const fps = 1000 / delta;
            this.fpsHistory.push(fps);

            // Keep last 30 samples
            if (this.fpsHistory.length > 30) {
                this.fpsHistory.shift();
            }

            // Calculate average FPS
            const avgFps = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
            this.elements.fpsDisplay.textContent = avgFps.toFixed(1);
        }
        this.lastFrameTime = now;
    }

    updateStats(stats, networkLatency) {
        // Encoder time
        if (stats.encoder_ms !== undefined) {
            this.elements.encoderTime.textContent = `${stats.encoder_ms.toFixed(1)} ms`;
        }

        // Decoder time
        if (stats.decoder_ms !== undefined) {
            this.elements.decoderTime.textContent = `${stats.decoder_ms.toFixed(1)} ms`;
        }

        // Gaussian count
        if (stats.num_gaussians !== undefined) {
            this.elements.gaussianCount.textContent = this.formatNumber(stats.num_gaussians);
        }

        // Total latency
        if (stats.total_latency_ms !== undefined) {
            this.elements.totalLatency.textContent = `${stats.total_latency_ms.toFixed(1)} ms`;
            this.elements.latencyDisplay.textContent = `${stats.total_latency_ms.toFixed(0)} ms`;
        }

        // Network latency
        this.elements.networkLatency.textContent = `${Math.max(0, networkLatency).toFixed(0)} ms`;

        // Per-column latency metrics
        if (stats.column_latency) {
            const cl = stats.column_latency;
            if (cl.input_ms !== undefined && this.elements.latencyInput) {
                this.elements.latencyInput.textContent = `${cl.input_ms.toFixed(0)}ms`;
            }
            if (cl.cropped_ms !== undefined && this.elements.latencyCropped) {
                this.elements.latencyCropped.textContent = `${cl.cropped_ms.toFixed(0)}ms`;
            }
            if (cl.gt_depth_ms !== undefined && this.elements.latencyGtDepth) {
                this.elements.latencyGtDepth.textContent = `${cl.gt_depth_ms.toFixed(0)}ms`;
            }
            if (cl.mono_depth_ms !== undefined && this.elements.latencyMono) {
                this.elements.latencyMono.textContent = `${cl.mono_depth_ms.toFixed(0)}ms`;
            }
            if (cl.predicted_depth_ms !== undefined && this.elements.latencyPredicted) {
                this.elements.latencyPredicted.textContent = `${cl.predicted_depth_ms.toFixed(0)}ms`;
            }
            if (cl.silhouette_ms !== undefined && this.elements.latencySilhouette) {
                this.elements.latencySilhouette.textContent = `${cl.silhouette_ms.toFixed(0)}ms`;
            }
        }

        // FPS from server
        if (stats.fps !== undefined) {
            // Could compare with client-side FPS
        }
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    updateConnectionStatus(status) {
        const statusEl = this.elements.connectionStatus;
        const dot = statusEl.querySelector('.status-dot');
        const text = statusEl.querySelector('span:last-child');

        switch (status) {
            case 'connected':
                dot.className = 'status-dot bg-green-500';
                text.textContent = 'Connected';
                text.className = 'text-sm text-green-400';
                break;
            case 'connecting':
                dot.className = 'status-dot bg-yellow-500';
                text.textContent = 'Connecting...';
                text.className = 'text-sm text-yellow-400';
                break;
            case 'disconnected':
                dot.className = 'status-dot bg-red-500';
                text.textContent = 'Disconnected';
                text.className = 'text-sm text-red-400';
                break;
        }
    }

    togglePause() {
        this.paused = !this.paused;
        this.elements.pauseBtn.textContent = this.paused ? 'Resume' : 'Pause';
        this.elements.pauseBtn.classList.toggle('bg-blue-600', this.paused);
        this.elements.pauseBtn.classList.toggle('hover:bg-blue-500', this.paused);
    }

    setViewMode(mode) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'set_view',
                mode: mode
            }));
            this.viewMode = mode;
        }
    }

    sendPing() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'ping',
                timestamp: Date.now()
            }));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Initialize viewer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.viewer = new DepthSplatViewer();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.viewer) {
        window.viewer.disconnect();
    }
});
