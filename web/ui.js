import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "GeminiImageLoader.ui",
  beforeRegisterNodeDef(nodeType, nodeData, appRef) {
    if (nodeData?.name !== "GeminiImageLoader") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);

      const widgets = this.widgets || [];
      const modeW = widgets.find((w) => w.name === "load_mode");
      const imageW = widgets.find((w) => w.name === "image");
      const videoW = widgets.find((w) => w.name === "video");
      const frameW = widgets.find((w) => w.name === "video_frame");
      const panelW = widgets.find((w) => w.name === "show_video_panel");

      // Always disable built-in upload overlays; we provide our own buttons
      if (imageW) {
        imageW.options = imageW.options || {};
        imageW.options.image_upload = false;
      }
      if (videoW) {
        videoW.options = videoW.options || {};
        videoW.options.video_upload = false;
        videoW.options.file_upload = false;
      }

      // Add our own "Choose Video" button to force a video/* file chooser and upload to input folder
      let chooseVideoBtn = this.addWidget?.("button", "Choose Video", null, async () => {
        try {
          const picker = document.createElement("input");
          picker.type = "file";
          picker.accept = "video/*";
          picker.onchange = async () => {
            const file = picker.files?.[0];
            if (!file) return;
            // Upload to Comfy /upload/image with type=input (server accepts any bytes)
            const fd = new FormData();
            fd.append("image", file, file.name);
            fd.append("type", "input");
            const res = await fetch("/upload/image", { method: "POST", body: fd });
            if (!res.ok) throw new Error(`upload failed: ${res.status}`);
            const json = await res.json();
            const saved = json?.name || file.name;
            if (videoW) {
              videoW.value = saved;
              videoW.callback?.(saved);
              setVideoSrc(saved);
            }
          };
          picker.click();
        } catch (e) {
          console.warn("[GeminiImageLoader] Choose Video failed", e);
        }
      }, { serialize: false });

      // (No custom Choose Image button; rely on built-in image upload in image mode)

      // Hidden video element used for frame capture (independent of panel)
      let hiddenVideo = document.createElement("video");
      hiddenVideo.muted = true;
      hiddenVideo.playsInline = true;
      hiddenVideo.loop = true;
      hiddenVideo.style.display = "none";
      document.body.appendChild(hiddenVideo);

      // Downscaled buffer canvas to minimize per-frame draw cost
      const bufCanvas = document.createElement("canvas");
      const bufCtx = bufCanvas.getContext("2d", { willReadFrequently: false });
      let capturing = false;
      let captureTimer = null;
      const MAX_BUFFER_WIDTH = 960; // upper cap; actual target matches preview window size
      // Image element fed into the built-in preview widget in video mode
      const previewImg = new Image();
      let lastPreviewUpdate = 0;
      const PREVIEW_UPDATE_MS = 66; // ~15 fps to balance clarity and smoothness

      const captureFrame = () => {
        if (!hiddenVideo || hiddenVideo.readyState < 2) return;
        const vw = hiddenVideo.videoWidth || 1;
        const vh = hiddenVideo.videoHeight || 1;
        // Compute preview window size below widgets so canvas matches widget's expected draw size
        const pad = 6;
        const widgetsY = (typeof this.widgets_start_y === "number" ? this.widgets_start_y : 18) + 4;
        const previewW = Math.max(64, this.size[0] - pad * 2);
        const previewH = Math.max(64, this.size[1] - widgetsY - pad);
        const ar = vw / vh;
        let fitW = previewW;
        let fitH = Math.floor(fitW / ar);
        if (fitH > previewH) { fitH = previewH; fitW = Math.floor(previewH * ar); }
        const targetW = Math.max(1, Math.min(MAX_BUFFER_WIDTH, vw, fitW));
        const targetH = Math.max(1, Math.round(targetW / ar));
        if (bufCanvas.width !== targetW || bufCanvas.height !== targetH) {
          bufCanvas.width = targetW;
          bufCanvas.height = targetH;
        }
        try {
          bufCtx.drawImage(hiddenVideo, 0, 0, targetW, targetH);
          // Update built-in preview image from buffer at a throttled rate
          const now = performance.now();
          if (modeW?.value === "video" && now - lastPreviewUpdate > PREVIEW_UPDATE_MS) {
            lastPreviewUpdate = now;
            try {
              previewImg.src = bufCanvas.toDataURL("image/webp", 0.7);
            } catch {
              previewImg.src = bufCanvas.toDataURL("image/png");
            }
          }
        } catch {}
      };

      const startCapture = () => {
        if (capturing) return;
        capturing = true;
        // Prefer requestVideoFrameCallback for smooth capture
        if (typeof hiddenVideo.requestVideoFrameCallback === "function") {
          const pump = () => {
            if (!capturing) return;
            captureFrame();
            this.setDirtyCanvas(true);
            hiddenVideo.requestVideoFrameCallback(pump);
          };
          hiddenVideo.requestVideoFrameCallback(pump);
        } else {
          // Fallback to ~30fps timer
          captureTimer = setInterval(() => {
            if (!capturing) return;
            captureFrame();
            this.setDirtyCanvas(true);
          }, 33);
        }
      };
      const stopCapture = () => {
        capturing = false;
        if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
      };

      // Optional floating video panel
      let panel = null;
      let videoEl = null;
      function ensurePanel() {
        if (panel) return panel;
        panel = document.getElementById("gemini-video-panel");
        if (!panel) {
          panel = document.createElement("div");
          panel.id = "gemini-video-panel";
          Object.assign(panel.style, {
            position: "fixed",
            right: "16px",
            bottom: "16px",
            width: "360px",
            background: "var(--comfy-menu-bg)",
            color: "var(--fg-color)",
            border: "1px solid var(--border-color)",
            borderRadius: "8px",
            boxShadow: "0 6px 24px rgba(0,0,0,.3)",
            padding: "8px",
            zIndex: 9999,
            display: "none",
          });
          const title = document.createElement("div");
          title.textContent = "Gemini Loader — Video Preview";
          Object.assign(title.style, { fontWeight: 600, marginBottom: "6px" });
          videoEl = document.createElement("video");
          videoEl.controls = true;
          videoEl.loop = true;
          videoEl.style.width = "100%";
          panel.appendChild(title);
          panel.appendChild(videoEl);
          document.body.appendChild(panel);
        }
        return panel;
      }

      function setVideoSrc(filename) {
        if (!filename) return;
        // Try ComfyUI /view endpoint for input files
        const url = `/view?filename=${encodeURIComponent(filename)}&type=input`;
        // Hidden video used for drawing
        try { hiddenVideo.pause(); } catch {}
        hiddenVideo.src = url;
        hiddenVideo.currentTime = 0;
        hiddenVideo.play().catch(()=>{});

        // Floating panel (optional)
        if (panelW?.value) {
          ensurePanel();
          videoEl.src = url;
          panel.style.display = "block";
        } else if (panel) {
          panel.style.display = "none";
        }
      }

      // Safe placeholder image used to suppress built-in image preview in video mode
      const placeholderImg = new Image();
      placeholderImg.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAuMBgqg3xVwAAAAASUVORK5CYII="; // 1x1 transparent
      placeholderImg.onload = () => { if (modeW?.value === "video") this.setDirtyCanvas(true); };

      // Hold previous image preview so we can restore after video mode
      if (!this.__gil_savedPreview) this.__gil_savedPreview = null;

      const updateVisibility = () => {
        const isVideo = modeW?.value === "video";
        if (imageW) {
          imageW.hidden = !!isVideo;
          imageW.options = imageW.options || {};
          // Enable the built-in image upload only in image mode
          imageW.options.image_upload = !isVideo;
        }
        if (videoW) {
          videoW.hidden = !isVideo;
          videoW.options = videoW.options || {};
          // Enable video upload only in video mode
          videoW.options.video_upload = !!isVideo;
          videoW.options.file_upload = !!isVideo;
        }
        if (chooseVideoBtn) chooseVideoBtn.hidden = !isVideo;
        if (frameW) frameW.hidden = !isVideo;
        // In video mode, feed the built-in preview with a throttled canvas image (smooth + correct sizing)
        if (isVideo) {
          if (!this.__gil_savedPreview) {
            this.__gil_savedPreview = { imgs: this.imgs, image: this.image };
          }
          this.imgs = [previewImg];
          this.image = previewImg;
        }
        // Ensure any active video capture/loop is stopped to keep UI responsive when leaving video mode
        if (!isVideo) {
          try { hiddenVideo.pause(); } catch {}
          stopCapture();
          if (panel) panel.style.display = "none";
          // Restore saved image preview if we previously suppressed it
          if (this.__gil_savedPreview) {
            this.imgs = this.__gil_savedPreview.imgs;
            this.image = this.__gil_savedPreview.image;
            this.__gil_savedPreview = null;
          }
        }
        if (isVideo && videoW && videoW.value) {
          setVideoSrc(videoW.value);
        } else if (panel) {
          panel.style.display = "none";
        }
        this.setDirtyCanvas(true, true);
      };


      // Overlay draw: render video into a fixed-height bottom window (matches typical image preview feel)
      const __oldDraw = this.onDrawForeground;
      this.onDrawForeground = function(ctx) {
        __oldDraw?.call(this, ctx);
        const isVideo = modeW?.value === "video";
        if (!(isVideo && hiddenVideo && hiddenVideo.readyState >= 2)) return;
        const pad = 6;
        const previewH = 140; // fixed window height for clarity
        const w = Math.max(64, this.size[0] - pad * 2);
        const h = previewH;
        const x = pad;
        const y = Math.max(24, this.size[1] - h - pad);

        const vw = hiddenVideo.videoWidth || 1;
        const vh = hiddenVideo.videoHeight || 1;
        const ar = vw / vh;
        let dw = w, dh = Math.floor(w / ar);
        if (dh > h) { dh = h; dw = Math.floor(h * ar); }
        const dx = x + Math.floor((w - dw) / 2);
        const dy = y + Math.floor((h - dh) / 2);

        const ctx2 = ctx;
        ctx2.save();
        ctx2.fillStyle = "#111a";
        ctx2.fillRect(x, y, w, h);
        try { ctx2.drawImage(hiddenVideo, dx, dy, dw, dh); } catch {}
        ctx2.strokeStyle = "#333";
        ctx2.strokeRect(x + .5, y + .5, w - 1, h - 1);
        ctx2.restore();
      };
      // Guard widget drawing (do not filter widgets here to avoid layout issues)
      const _origDrawWidgets = this.drawWidgets;
      this.drawWidgets = function () {
        try {
          return _origDrawWidgets?.apply(this, arguments);
        } catch (e) {
          console.warn("[GeminiImageLoader] widget draw error", e);
          return 0;
        }
      };

      if (modeW) {
        const cb = modeW.callback;
        modeW.callback = (v) => {
          cb?.(v);
          updateVisibility();
        };
      }

      if (videoW) {
        const old = videoW.callback;
        videoW.callback = (v) => {
          old?.(v);
          if (modeW?.value === "video" && v) setVideoSrc(v);
        };
      }

      if (panelW) {
        const old = panelW.callback;
        panelW.callback = (v) => {
          old?.(v);
          if (!panel) return;
          panel.style.display = v ? "block" : "none";
        };
      }

      // Initial state
      updateVisibility();

      // Simple redraw helper (used by event hooks)
      const requestRedraw = () => { this.setDirtyCanvas(true); };

      // Wire hidden video events to request redraw (throttled)
      hiddenVideo.addEventListener("play", () => { startCapture(); requestRedraw(); });
      hiddenVideo.addEventListener("pause", () => { stopCapture(); requestRedraw(); });
      hiddenVideo.addEventListener("ended", () => { stopCapture(); requestRedraw(); });
      hiddenVideo.addEventListener("seeked", () => { captureFrame(); requestRedraw(); });
      hiddenVideo.addEventListener("timeupdate", () => { captureFrame(); requestRedraw(); });

      // Cleanup on node removal to prevent leaks
      const oldOnRemoved = this.onRemoved;
      this.onRemoved = function() {
        try {
          stopCapture();
          hiddenVideo.pause();
          hiddenVideo.src = "";
          hiddenVideo.remove();
          if (panel) panel.remove();
        } catch {}
        oldOnRemoved?.call(this);
      };
    };
  },
});
