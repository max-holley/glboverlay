import { formatBytes, type ModelMeta } from './storage';
import type { GlobalSettings, ModelMode, ModelSettings } from './settings';

type El<K extends keyof HTMLElementTagNameMap> = HTMLElementTagNameMap[K];

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...kids: (Node | string)[]
): El<K> {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea, [tabindex]:not([tabindex="-1"])';

/** Overlay panel with focus management. */
class Panel {
  readonly el: HTMLElement;
  readonly body: HTMLElement;
  private prev: HTMLElement | null = null;
  isOpen = false;

  constructor(
    id: string,
    title: string,
    private onClose: () => void,
    wide = false,
  ) {
    this.body = h('div', { class: 'panel-body' });
    const close = h('button', { class: 'icon-btn', 'aria-label': `Close ${title}`, type: 'button' }, '✕');
    close.addEventListener('click', () => this.onClose());
    this.el = h(
      'section',
      { class: `panel${wide ? ' wide' : ''}`, id, role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': `${id}-title`, tabindex: '-1', hidden: '' },
      h('header', { class: 'panel-head' }, h('h2', { id: `${id}-title` }, title), close),
      this.body,
    );
    this.el.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab') return;
      const f = [...this.el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.offsetParent !== null);
      if (f.length === 0) return;
      const first = f[0];
      const last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        last.focus();
        e.preventDefault();
      } else if (!e.shiftKey && document.activeElement === last) {
        first.focus();
        e.preventDefault();
      }
    });
  }

  open(): void {
    if (this.isOpen) return;
    this.prev = document.activeElement as HTMLElement | null;
    this.el.hidden = false;
    // next frame so the transition runs
    requestAnimationFrame(() => this.el.classList.add('open'));
    this.isOpen = true;
    // Focus the panel itself so global shortcuts keep working; Tab moves into the controls.
    this.el.focus({ preventScroll: true });
  }

  close(): void {
    if (!this.isOpen) return;
    this.isOpen = false;
    this.el.classList.remove('open');
    const done = () => {
      if (!this.isOpen) this.el.hidden = true;
    };
    setTimeout(done, 200);
    if (this.prev && document.contains(this.prev)) this.prev.focus({ preventScroll: true });
    else (document.activeElement as HTMLElement | null)?.blur();
  }
}

export interface UIHandlers {
  addFiles(files: File[]): void;
  selectModel(id: string): void;
  renameModel(id: string, name: string): void;
  deleteModel(id: string): void;
  globalChanged(restartCamera: boolean): void;
  modelSettingsChanged(): void;
  resetGlobal(): void;
  resetModel(): void;
  recalibrate(): void;
  retryCamera(): void;
}

interface Range {
  key: 'smoothing' | 'expression' | 'headSensitivity';
  label: string;
  min: number;
  max: number;
  step: number;
}

const RANGES: Range[] = [
  { key: 'smoothing', label: 'Smoothing', min: 0, max: 1, step: 0.01 },
  { key: 'expression', label: 'Expression intensity', min: 0, max: 2, step: 0.01 },
  { key: 'headSensitivity', label: 'Head-motion sensitivity', min: 0, max: 2, step: 0.01 },
];

const NUMS: { key: 'px' | 'py' | 'pz' | 'scale' | 'rx' | 'ry' | 'rz'; label: string; step: string }[] = [
  { key: 'px', label: 'Position X', step: '0.01' },
  { key: 'py', label: 'Position Y', step: '0.01' },
  { key: 'pz', label: 'Position Z', step: '0.01' },
  { key: 'scale', label: 'Scale', step: '0.01' },
  { key: 'rx', label: 'Rotate X°', step: '1' },
  { key: 'ry', label: 'Rotate Y°', step: '1' },
  { key: 'rz', label: 'Rotate Z°', step: '1' },
];

export class UI {
  readonly info: Panel;
  readonly models: Panel;
  readonly settings: Panel;
  private panels: Panel[];

  private dropzone: HTMLElement;
  private dropOverlay: HTMLElement;
  private toastEl: HTMLElement | null = null;
  private toastTimer = 0;
  private errorEl: HTMLElement;
  private loadingEl: HTMLElement;
  private banner: HTMLElement;
  private fileInput: HTMLInputElement;
  private grid: HTMLElement;
  private storageText: HTMLElement;
  private storageBar: HTMLElement;
  private dialogEl: HTMLElement;

  // settings controls
  private camSel!: HTMLSelectElement;
  private resSel!: HTMLSelectElement;
  private fpsSel!: HTMLSelectElement;
  private trackSel!: HTMLSelectElement;
  private renderSel!: HTMLSelectElement;
  private qualitySel!: HTMLSelectElement;
  private mirror!: HTMLInputElement;
  private ranges = new Map<string, { input: HTMLInputElement; out: HTMLOutputElement }>();
  private modelSection!: HTMLElement;
  private modeSel!: HTMLSelectElement;
  private nums = new Map<string, HTMLInputElement>();
  private flips = new Map<string, HTMLInputElement>();
  private modeNote!: HTMLElement;
  private trackerLine!: HTMLElement;

  private getGlobal: () => GlobalSettings;
  private getModelSettings: () => ModelSettings | null;

  constructor(
    root: HTMLElement,
    private hd: UIHandlers,
    getGlobal: () => GlobalSettings,
    getModelSettings: () => ModelSettings | null,
  ) {
    this.getGlobal = getGlobal;
    this.getModelSettings = getModelSettings;
    const closeAll = () => this.closePanels();
    this.info = new Panel('info', 'glboverlay', closeAll);
    this.models = new Panel('models', 'Models', closeAll, true);
    this.settings = new Panel('settings', 'Settings', closeAll);
    this.panels = [this.info, this.models, this.settings];

    this.fileInput = h('input', {
      type: 'file',
      multiple: '',
      accept: '.glb,.gltf,.vrm,.bin,.png,.jpg,.jpeg,.webp,.ktx2',
      class: 'sr-only',
      tabindex: '-1',
    });
    this.fileInput.addEventListener('change', () => {
      const files = [...(this.fileInput.files ?? [])];
      this.fileInput.value = '';
      if (files.length) this.hd.addFiles(files);
    });

    const choose = h('button', { class: 'btn primary', type: 'button' }, 'Choose file');
    choose.addEventListener('click', () => this.fileInput.click());
    this.dropzone = h(
      'div',
      { class: 'dropzone', hidden: '' },
      h(
        'div',
        { class: 'dz-card' },
        h('h1', {}, 'glboverlay'),
        h('p', {}, 'Drop a 3D model anywhere to map it to your face.'),
        h('p', { class: 'muted' }, '.glb · .gltf (with .bin and textures) · .vrm'),
        choose,
      ),
    );
    this.dropOverlay = h('div', { class: 'drop-overlay', hidden: '' }, h('div', {}, 'Drop to add model'));
    this.errorEl = h('div', { class: 'error-card', role: 'alert', hidden: '' });
    this.loadingEl = h('div', { class: 'loading', role: 'status', hidden: '' });
    this.banner = h('div', { class: 'banner', hidden: '' });
    this.dialogEl = h('div', { class: 'dialog-wrap', hidden: '' });

    this.grid = h('div', { class: 'grid' });
    this.storageText = h('span', {});
    this.storageBar = h('div', { class: 'bar-fill' });

    this.buildInfo();
    this.buildModels();
    this.buildSettings();

    root.append(
      this.dropzone,
      this.banner,
      this.errorEl,
      this.loadingEl,
      ...this.panels.map((p) => p.el),
      this.dialogEl,
      this.dropOverlay,
      this.fileInput,
    );
    this.bindDrop();
  }

  // ---------- panels ----------

  closePanels(): boolean {
    const was = this.panels.some((p) => p.isOpen);
    this.panels.forEach((p) => p.close());
    return was;
  }

  toggle(p: Panel): void {
    const wasOpen = p.isOpen;
    this.panels.forEach((x) => x !== p && x.close());
    if (wasOpen) p.close();
    else {
      if (p === this.settings) this.syncSettings();
      p.open();
    }
  }

  get anyOpen(): boolean {
    return this.panels.some((p) => p.isOpen) || !this.dialogEl.hidden;
  }

  // ---------- info ----------

  private buildInfo(): void {
    const b = this.info.body;
    b.innerHTML = `
      <p>Maps a 3D model to your head and face using your webcam, entirely in your browser.
      Capture this window in OBS with <b>Window Capture</b> and a <b>Chroma Key</b> filter.</p>
      <h3>Keyboard</h3>
      <dl class="keys">
        <dt><kbd>C</kbd></dt><dd>Toggle green-screen background (<code>#00FF00</code>) and hide all UI</dd>
        <dt><kbd>I</kbd></dt><dd>This panel</dd>
        <dt><kbd>M</kbd></dt><dd>Model manager</dd>
        <dt><kbd>S</kbd></dt><dd>Settings</dd>
        <dt><kbd>R</kbd></dt><dd>Recalibrate: capture your current pose as neutral</dd>
        <dt><kbd>Esc</kbd></dt><dd>Close panels</dd>
      </dl>
      <h3>Models</h3>
      <p>Supported: <code>.glb</code>, <code>.gltf</code> (select or drop it together with its <code>.bin</code> and textures) and <code>.vrm</code>
      (0.x and 1.0). Draco, Meshopt and KTX2 compression work.</p>
      <ul>
        <li><b>Best facial animation:</b> VRM models, Perfect Sync VRMs, or models with ARKit-52 morph targets (e.g. Ready Player Me).</li>
        <li><b>Avatars</b> (humanoid skeleton or VRM) move their head, neck, eyes, jaw and expressions.</li>
        <li><b>Anything else</b> (a motorcycle, a helmet) is moved as a rigid object following your head. Tune offsets in Settings.</li>
        <li>Keep files under 100 MB; above 250 MB they are rejected. Compress with Draco or Meshopt if needed.</li>
      </ul>
      <h3>OBS setup</h3>
      <ol>
        <li>Add a <b>Window Capture</b> source and choose this Chrome window. Keep it visible, not minimised: Chrome pauses rendering for minimised or fully covered windows.</li>
        <li>Press <kbd>C</kbd> for the green background, which also hides the UI and cursor.</li>
        <li>Right-click the source, <b>Filters</b>, <b>Crop/Pad</b> to the render area (remove the tab strip and address bar).</li>
        <li>Add a <b>Chroma Key</b> filter: key colour <b>Green</b> (or custom <code>#00FF00</code>), similarity about <b>400</b>, smoothness about <b>80</b>, key colour spill reduction about <b>100</b>. Lower similarity if parts of the model vanish.</li>
      </ol>
      <h3>Privacy</h3>
      <p>Everything runs locally. The webcam feed is never shown, recorded or uploaded; models are stored only in this browser (IndexedDB). No accounts, no analytics.</p>`;
  }

  // ---------- models ----------

  private buildModels(): void {
    const add = h('button', { class: 'btn primary', type: 'button' }, 'Add model');
    add.addEventListener('click', () => this.fileInput.click());
    const bar = h('div', { class: 'bar' }, this.storageBar);
    this.models.body.append(
      h('div', { class: 'row between' }, h('p', { class: 'muted' }, 'Drag and drop files anywhere to add a model.'), add),
      this.grid,
      h('div', { class: 'storage' }, bar, this.storageText),
    );
  }

  setModels(list: ModelMeta[], activeId: string | null): void {
    this.grid.replaceChildren();
    if (list.length === 0) this.grid.append(h('p', { class: 'muted' }, 'No stored models yet.'));
    for (const m of list) {
      const thumb = h('div', { class: 'thumb' });
      if (m.thumbnail) {
        const img = h('img', { alt: '', draggable: 'false' });
        img.src = URL.createObjectURL(m.thumbnail);
        img.addEventListener('load', () => URL.revokeObjectURL(img.src), { once: true });
        thumb.append(img);
      } else thumb.append(h('span', { class: 'muted' }, '3D'));

      const name = h('div', { class: 'name', title: m.name }, m.name);
      const sel = h('button', { class: 'btn small primary', type: 'button' }, m.id === activeId ? 'In use' : 'Select');
      if (m.id === activeId) sel.setAttribute('aria-pressed', 'true');
      sel.addEventListener('click', () => this.hd.selectModel(m.id));
      const ren = h('button', { class: 'btn small', type: 'button' }, 'Rename');
      ren.addEventListener('click', () => {
        const input = h('input', { class: 'rename', type: 'text', value: m.name, 'aria-label': 'Model name', maxlength: '80' });
        name.replaceWith(input);
        input.focus();
        input.select();
        let finished = false;
        const finish = (save: boolean) => {
          if (finished) return;
          finished = true;
          const v = input.value.trim();
          if (save && v && v !== m.name) this.hd.renameModel(m.id, v);
          else input.replaceWith(name);
        };
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') finish(true);
          else if (e.key === 'Escape') {
            e.stopPropagation();
            finish(false);
          }
        });
        input.addEventListener('blur', () => finish(true));
      });
      const del = h('button', { class: 'btn small danger', type: 'button' }, 'Delete');
      del.addEventListener('click', () => this.hd.deleteModel(m.id));

      this.grid.append(
        h(
          'article',
          { class: `card${m.id === activeId ? ' active' : ''}` },
          thumb,
          name,
          h('div', { class: 'meta muted' }, `${formatBytes(m.size)} · ${new Date(m.date).toLocaleDateString()}`),
          h('div', { class: 'row gap' }, sel, ren, del),
        ),
      );
    }
  }

  setStorage(e: { usage: number; quota: number } | null, persisted: boolean): void {
    if (!e) {
      this.storageText.textContent = 'Storage usage unavailable.';
      this.storageBar.style.width = '0';
      return;
    }
    this.storageBar.style.width = `${Math.min(100, (e.usage / e.quota) * 100).toFixed(2)}%`;
    this.storageText.textContent = `${formatBytes(e.usage)} used of ${formatBytes(e.quota)} available${persisted ? ' · persistent' : ' · not persistent (browser may evict)'}`;
  }

  // ---------- settings ----------

  private buildSettings(): void {
    const b = this.settings.body;
    const field = (label: string, control: HTMLElement) => h('label', { class: 'field' }, h('span', {}, label), control);

    this.trackerLine = h('p', { class: 'muted small-text', role: 'status' }, 'Tracking: not running');
    this.camSel = h('select', {});
    this.resSel = h('select', {});
    for (const [w, hh] of [
      [640, 480],
      [1280, 720],
      [1920, 1080],
    ])
      this.resSel.append(h('option', { value: `${w}x${hh}` }, `${w}×${hh}`));
    this.fpsSel = h('select', {});
    for (const f of [15, 24, 30, 60]) this.fpsSel.append(h('option', { value: String(f) }, `${f} fps`));
    this.mirror = h('input', { type: 'checkbox' });
    this.trackSel = h('select', {});
    for (const f of [10, 15, 20, 30]) this.trackSel.append(h('option', { value: String(f) }, `${f} fps`));
    this.renderSel = h('select', {});
    for (const f of [24, 30, 60]) this.renderSel.append(h('option', { value: String(f) }, `${f} fps`));
    this.renderSel.append(h('option', { value: '0' }, 'Display refresh'));
    this.qualitySel = h('select', {});
    for (const [q, label] of [
      [0.5, 'Low (0.5x)'],
      [0.75, 'Medium (0.75x)'],
      [1, 'Standard (1x)'],
      [1.5, 'High (1.5x)'],
      [2, 'Max (2x)'],
    ] as const)
      this.qualitySel.append(h('option', { value: String(q) }, label));
    const perfChange = () => {
      const g = this.getGlobal();
      g.trackFps = Number(this.trackSel.value);
      g.renderFps = Number(this.renderSel.value);
      g.quality = Number(this.qualitySel.value);
      this.hd.globalChanged(false);
    };
    this.trackSel.addEventListener('change', perfChange);
    this.renderSel.addEventListener('change', perfChange);
    this.qualitySel.addEventListener('change', perfChange);

    const camChange = () => {
      const g = this.getGlobal();
      g.camera.deviceId = this.camSel.value;
      const [w, hh] = this.resSel.value.split('x').map(Number);
      g.camera.width = w;
      g.camera.height = hh;
      g.camera.fps = Number(this.fpsSel.value);
      this.hd.globalChanged(true);
    };
    this.camSel.addEventListener('change', camChange);
    this.resSel.addEventListener('change', camChange);
    this.fpsSel.addEventListener('change', camChange);
    this.mirror.addEventListener('change', () => {
      this.getGlobal().mirror = this.mirror.checked;
      this.hd.globalChanged(false);
    });

    b.append(
      this.trackerLine,
      h('h3', {}, 'Camera'),
      field('Device', this.camSel),
      field('Resolution', this.resSel),
      field('Frame rate', this.fpsSel),
      field('Mirror', this.mirror),
      h('h3', {}, 'Performance'),
      field('Tracking rate', this.trackSel),
      field('Render rate', this.renderSel),
      field('Render quality', this.qualitySel),
      h('h3', {}, 'Tracking'),
    );
    for (const r of RANGES) {
      const input = h('input', { type: 'range', min: String(r.min), max: String(r.max), step: String(r.step) });
      const out = h('output', {});
      input.addEventListener('input', () => {
        this.getGlobal()[r.key] = Number(input.value);
        out.textContent = Number(input.value).toFixed(2);
        this.hd.globalChanged(false);
      });
      this.ranges.set(r.key, { input, out });
      b.append(h('label', { class: 'field' }, h('span', {}, r.label), h('div', { class: 'range' }, input, out)));
    }
    const reset = h('button', { class: 'btn', type: 'button' }, 'Reset to defaults');
    reset.addEventListener('click', () => {
      this.hd.resetGlobal();
      this.syncSettings();
    });
    b.append(h('div', { class: 'row' }, reset));

    // per-model
    this.modeSel = h('select', {});
    for (const [v, l] of [
      ['auto', 'Auto-detect'],
      ['avatar', 'Avatar (head, face, bones)'],
      ['rigid', 'Rigid (move whole model)'],
    ])
      this.modeSel.append(h('option', { value: v }, l));
    this.modeNote = h('p', { class: 'muted small-text' });
    this.modeSel.addEventListener('change', () => {
      const m = this.getModelSettings();
      if (!m) return;
      m.mode = this.modeSel.value as ModelMode;
      this.hd.modelSettingsChanged();
    });
    this.modelSection = h('div', {}, h('h3', {}, 'Current model'), field('Mode', this.modeSel), this.modeNote);
    const grid = h('div', { class: 'num-grid' });
    for (const n of NUMS) {
      const input = h('input', { type: 'number', step: n.step });
      input.addEventListener('input', () => {
        const m = this.getModelSettings();
        const v = Number(input.value);
        if (!m || !Number.isFinite(v)) return;
        m[n.key] = n.key === 'scale' ? Math.max(0.01, v) : v;
        this.hd.modelSettingsChanged();
      });
      this.nums.set(n.key, input);
      grid.append(field(n.label, input));
    }
    const flipRow = h('div', { class: 'row gap' });
    for (const k of ['flipX', 'flipY', 'flipZ'] as const) {
      const cb = h('input', { type: 'checkbox' });
      cb.addEventListener('change', () => {
        const m = this.getModelSettings();
        if (!m) return;
        m[k] = cb.checked;
        this.hd.modelSettingsChanged();
      });
      this.flips.set(k, cb);
      flipRow.append(h('label', { class: 'check' }, cb, `Flip ${k.slice(-1)}`));
    }
    const recal = h('button', { class: 'btn', type: 'button' }, 'Recalibrate (R)');
    recal.addEventListener('click', () => this.hd.recalibrate());
    const resetM = h('button', { class: 'btn', type: 'button' }, 'Reset model calibration');
    resetM.addEventListener('click', () => {
      this.hd.resetModel();
      this.syncSettings();
    });
    this.modelSection.append(grid, flipRow, h('div', { class: 'row gap' }, recal, resetM));
    b.append(this.modelSection);
  }

  syncSettings(): void {
    const g = this.getGlobal();
    this.resSel.value = `${g.camera.width}x${g.camera.height}`;
    if (!this.resSel.value) this.resSel.value = '1280x720';
    this.fpsSel.value = String(g.camera.fps);
    this.camSel.value = g.camera.deviceId;
    this.mirror.checked = g.mirror;
    this.trackSel.value = String(g.trackFps);
    this.renderSel.value = String(g.renderFps);
    this.qualitySel.value = String(g.quality);
    for (const r of RANGES) {
      const c = this.ranges.get(r.key)!;
      c.input.value = String(g[r.key]);
      c.out.textContent = g[r.key].toFixed(2);
    }
    const m = this.getModelSettings();
    this.modelSection.hidden = !m;
    if (m) {
      this.modeSel.value = m.mode;
      for (const n of NUMS) this.nums.get(n.key)!.value = String(m[n.key]);
      for (const k of ['flipX', 'flipY', 'flipZ'] as const) this.flips.get(k)!.checked = m[k];
    }
  }

  setModeNote(text: string): void {
    this.modeNote.textContent = text;
  }

  setDevices(devices: MediaDeviceInfo[]): void {
    const g = this.getGlobal();
    this.camSel.replaceChildren(h('option', { value: '' }, 'Default camera'));
    devices.forEach((d, i) => this.camSel.append(h('option', { value: d.deviceId }, d.label || `Camera ${i + 1}`)));
    this.camSel.value = g.camera.deviceId;
    if (this.camSel.value !== g.camera.deviceId) this.camSel.value = '';
  }

  private trackerText = 'Tracking: not running';
  private renderFps = 0;

  setTrackerStatus(s: { running: boolean; delegate: string; fps: number }): void {
    this.trackerText = s.running ? `Tracking: ${s.delegate} · ${s.fps.toFixed(0)} fps` : 'Tracking: not running';
    this.updateTrackerLine();
  }

  setRenderFps(fps: number): void {
    if (Math.round(fps) === Math.round(this.renderFps)) return;
    this.renderFps = fps;
    this.updateTrackerLine();
  }

  private updateTrackerLine(): void {
    this.trackerLine.textContent = `${this.trackerText} · render ${this.renderFps.toFixed(0)} fps`;
  }

  // ---------- transient overlays ----------

  showToast(text: string): void {
    this.toastEl?.remove();
    clearTimeout(this.toastTimer);
    const x = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Dismiss' }, '✕');
    const t = h('div', { class: 'toast', role: 'status' }, h('span', {}, text), x);
    const dismiss = () => {
      t.classList.remove('show');
      setTimeout(() => t.remove(), 400);
      clearTimeout(this.toastTimer);
    };
    t.addEventListener('click', dismiss);
    document.getElementById('ui')!.append(t);
    requestAnimationFrame(() => t.classList.add('show'));
    this.toastEl = t;
    this.toastTimer = window.setTimeout(dismiss, 6000);
  }

  showError(message: string, action?: { label: string; run: () => void }): void {
    this.errorEl.replaceChildren(h('p', {}, message));
    if (action) {
      const b = h('button', { class: 'btn small', type: 'button' }, action.label);
      b.addEventListener('click', action.run);
      this.errorEl.append(b);
    }
    const x = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Dismiss error' }, '✕');
    x.addEventListener('click', () => this.clearError());
    this.errorEl.append(x);
    this.errorEl.hidden = false;
  }

  clearError(): void {
    this.errorEl.hidden = true;
  }

  showBanner(text: string): void {
    const x = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Dismiss' }, '✕');
    x.addEventListener('click', () => (this.banner.hidden = true));
    this.banner.replaceChildren(h('span', {}, text), x);
    this.banner.hidden = false;
  }

  setLoading(text: string | null): void {
    this.loadingEl.hidden = text === null;
    if (text !== null) this.loadingEl.textContent = text;
  }

  showDropzone(show: boolean): void {
    this.dropzone.hidden = !show;
  }

  /** Fatal full-screen state (e.g. WebGL unavailable). */
  fatal(message: string): void {
    this.dropzone.hidden = true;
    this.errorEl.className = 'error-card fatal';
    this.errorEl.replaceChildren(h('h2', {}, 'glboverlay cannot start'), h('p', {}, message));
    this.errorEl.hidden = false;
  }

  confirm(message: string, okLabel: string, danger = false): Promise<boolean> {
    return new Promise((resolve) => {
      const prev = document.activeElement as HTMLElement | null;
      const ok = h('button', { class: `btn ${danger ? 'danger' : 'primary'}`, type: 'button' }, okLabel);
      const cancel = h('button', { class: 'btn', type: 'button' }, 'Cancel');
      const box = h(
        'div',
        { class: 'dialog', role: 'alertdialog', 'aria-modal': 'true', 'aria-label': message, tabindex: '-1' },
        h('p', {}, message),
        h('div', { class: 'row gap end' }, cancel, ok),
      );
      const finish = (v: boolean) => {
        this.dialogEl.hidden = true;
        this.dialogEl.replaceChildren();
        prev?.focus({ preventScroll: true });
        resolve(v);
      };
      ok.addEventListener('click', () => finish(true));
      cancel.addEventListener('click', () => finish(false));
      box.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          e.stopPropagation();
          finish(false);
        } else if (e.key === 'Tab') {
          e.preventDefault();
          (document.activeElement === ok ? cancel : ok).focus();
        }
      });
      this.dialogEl.replaceChildren(box);
      this.dialogEl.hidden = false;
      cancel.focus();
    });
  }

  // ---------- drag and drop ----------

  private bindDrop(): void {
    let depth = 0;
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files');
    window.addEventListener('dragenter', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth++;
      this.dropOverlay.hidden = false;
    });
    window.addEventListener('dragover', (e) => {
      if (hasFiles(e)) e.preventDefault();
    });
    window.addEventListener('dragleave', (e) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) this.dropOverlay.hidden = true;
    });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      this.dropOverlay.hidden = true;
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) this.hd.addFiles(files);
    });
  }
}
