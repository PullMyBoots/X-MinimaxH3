const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
const tr = value => window.H3I18n?.t(String(value)) || String(value);
let options = null;
let jobs = [];
const selectedHistoryJobIds = new Set();
let videoObjectUrl = null;
let currentEngine = null;
let currentLauncher = null;
let draggedJobId = null;
let activePage = ['create','infinite','tasks'].includes(new URLSearchParams(location.search).get('page'))
  ? new URLSearchParams(location.search).get('page') : 'tasks';
let engineReconcilePromise = null;
let workspaceBrowseParent = null;
let bootReady = false;
let uiPollPromise = null;
const CREATION_RESOLUTION_DETENTS = [360, 480, 540, 720, 900, 1080, 1220, 1440];
const FIRST_PASS_RESOLUTION_DETENTS = [360, 480, 540, 720, 900, 1080];
let globalLoraPolicy = {selected:null, loaded:null, changing:false, available:[]};
let globalFaceRepairPolicy = {canvas_size:768, capacity:4};
let globalPreviewPolicy = {steps:2};
let globalSecondSamplingWindowPolicy = {enabled:true, window_seconds:5.0, overlap_seconds:1.0};

function currentVariant() { return $('[name="model_variant"]')?.value || 'base'; }
function syncModelVariantToggle() {
  const toggle = $('#loraAccelerationEnabled');
  const summary = $('#modelVariantSummary');
  if (!toggle || !summary) return;
  const lora = currentVariant() === 'lora';
  toggle.checked = lora;
  const english = window.H3I18n?.locale === 'en';
  const baseTier = options?.active_weight_tier === 'w4a8' ? 'W4A8' : 'INT8';
  const activeLoRA = (globalLoraPolicy?.available || []).find(
    item => item.id === globalLoraPolicy?.selected
  );
  const loraName = activeLoRA?.profile?.display_name || 'LoRA Turbo';
  summary.textContent = lora
    ? `${english ? 'On' : '开启'} · ${loraName}`
    : `${english ? 'Off' : '关闭'} · ${baseTier} ${english ? 'base weights' : '原始权重'}`;
}
function currentEngineKey() {
  if (currentEngine === 'reference') return currentVariant() === 'lora' ? 'reference_lora' : 'reference';
  return currentVariant() === 'lora' ? 'lora' : 'original';
}

function updateSubmitAvailability() {
  const button = $('.submit-button', $('#generationForm'));
  if (!button || button.dataset.submitting === 'true') return;
  button.disabled = !bootReady || !currentEngine || Boolean(options?.engine_control?.switching);
  button.title = !bootReady
    ? '控制台正在初始化'
    : !currentEngine
      ? '请先选择生成模式'
      : options?.engine_control?.switching
        ? '模型正在切换，请稍候'
        : '发送生成任务';
}

function renderEngineLobby() {
  const unified = options?.deployment_mode === 'unified_console';
  const active = options?.current_engine || null;
  $('#engineLobby').hidden = !unified || Boolean(active);
  $('#createPage').hidden = unified && !active;
  $('#infinitePage').hidden = unified && !active;
  if (unified && !active) $('#tasksPage').hidden = true;
  $('.workspace-tabs').hidden = unified && !active;
  $('#exitEngine').hidden = !unified || !active;
  if (active) {
    $('#engineLobby').classList.remove('loading');
    $('#engineLobbyMessage').hidden = true;
  }
  if (!unified) return;
  renderWorkspace();
  const choices = options.model_choices || {};
  const ranges = options.host_memory?.budget_ranges || {};
  const groups = [['w4a8', 'W4A8 轻量权重'], ['int8', 'INT8 高质量权重']];
  $('#engineChoices').innerHTML = groups.map(([weight, title]) => {
    const range = ranges[weight] || {available:false, reason:'当前主机内存不足'};
    const minimum = Number(range.minimum_gib || (weight === 'w4a8' ? 12 : 24));
    const maximum = Number(range.maximum_gib || minimum);
    const stored = Number(localStorage.getItem(`h3serve_host_limit_${weight}`));
    const selected = Math.max(minimum, Math.min(maximum, stored || Number(range.recommended_gib) || minimum));
    const weightChoices = Object.entries(choices).filter(([, info]) => info.weight_tier === weight);
    return `<section class="engine-choice-group ${range.available ? '' : 'unavailable'}">
      <h2>${escapeHtml(title)}</h2>
      <label class="engine-memory-budget"><span>分配给H3服务的内存硬上限 <output data-memory-output="${weight}">${selected} GiB</output></span>
        <input type="range" min="${minimum}" max="${maximum}" step="1" value="${selected}" data-memory-weight="${weight}" ${range.available ? '' : 'disabled'}>
        <small>${range.available ? `不会预占 · 系统保留6GiB · 可选 ${minimum}–${maximum}GiB` : escapeHtml(range.reason || '不可用')}</small>
      </label>
      <div>${weightChoices.map(([key, info]) => `
        <button type="button" class="engine-choice" data-enter-engine="${escapeHtml(key)}" ${range.available ? '' : 'disabled'}>
          <strong>${escapeHtml(info.label)}</strong><small>${escapeHtml(info.description || '')}</small>
        </button>`).join('')}</div></section>`;
  }).join('');
  $$('[data-enter-engine]').forEach(button => button.addEventListener('click', () => enterEngine(button.dataset.enterEngine)));
  $$('[data-memory-weight]').forEach(input => input.addEventListener('input', () => {
    const weight = input.dataset.memoryWeight;
    $(`[data-memory-output="${weight}"]`).textContent = `${input.value} GiB`;
    localStorage.setItem(`h3serve_host_limit_${weight}`, input.value);
  }));
  renderEngineLoadProgress(options?.warm_state, options?.engine_control?.switching);
}

function renderEngineLoadProgress(warmState, switching=false) {
  const panel = $('#engineLoadProgress');
  if (!panel) return;
  const warm = warmState || {};
  const visible = Boolean(switching) || warm.status === 'loading';
  panel.hidden = !visible;
  if (!visible) return;
  const percent = Math.max(1, Math.min(99, Number(warm.progress_percent) || 1));
  const stageNames = {
    starting:'启动模型加载', preflight:'检查运行环境', model_paths:'准备本地权重',
    text_encoder:'准备文本编码器', model_graphs:'装配模型组件',
    vae_warmup:'编译预热视频VAE', host_memory:'整理主机内存',
    finalize:'完成运行时初始化',
  };
  $('#engineLoadStage').textContent = tr(stageNames[warm.progress_stage] || '正在加载模型引擎');
  $('#engineLoadPercent').textContent = `${percent.toFixed(0)}%`;
  $('#engineLoadBar').value = percent;
  $('#engineLoadDetail').textContent = tr(warm.progress_detail || '首次加载需要读取并装配模型权重');
}

function renderWorkspace() {
  const workspace = options?.workspace?.current;
  if (!workspace) return;
  $('#workspaceName').textContent = workspace.is_default ? '默认工作空间' : (workspace.name || '工作空间');
  $('#workspacePath').textContent = workspace.path;
  $('#workspacePath').title = workspace.path;
  $('#chooseWorkspace').disabled = !options.workspace.switchable;
}

async function browseWorkspace(path) {
  const suffix = path ? `?path=${encodeURIComponent(path)}` : '';
  const document = await (await api(`/api/v1/workspace/browse${suffix}`)).json();
  $('#workspacePathInput').value = document.path;
  workspaceBrowseParent = document.parent;
  $('#workspaceParent').disabled = !document.parent;
  $('#workspaceDirectories').innerHTML = document.directories.length
    ? document.directories.map(item => `<button type="button" class="workspace-directory" data-workspace-path="${escapeHtml(item.path)}"><i>▸</i><span>${escapeHtml(item.name)}</span></button>`).join('')
    : '<div class="manager-empty">这个文件夹中没有子文件夹</div>';
  $$('.workspace-directory').forEach(button => button.addEventListener('click', () => browseWorkspace(button.dataset.workspacePath).catch(showWorkspaceError)));
}

function showWorkspaceError(error) {
  $('#workspaceMessage').textContent = error.message;
  $('#workspaceMessage').hidden = false;
}

async function openWorkspaceDialog() {
  $('#workspaceMessage').hidden = true;
  $('#workspaceDialog').showModal();
  try { await browseWorkspace(options.workspace.current.path); }
  catch (error) { showWorkspaceError(error); }
}

async function activateWorkspace() {
  const button = $('#selectWorkspace');
  button.disabled = true; button.textContent = '正在切换…';
  try {
    await api('/api/v1/workspace', {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:$('#workspacePathInput').value.trim()}),
    });
    $('#workspaceDialog').close();
    jobs = [];
    await reloadOptions();
    await refreshJobs();
  } catch (error) { showWorkspaceError(error); }
  finally { button.disabled = false; button.textContent = '使用这个文件夹'; }
}

async function enterEngine(choiceKey) {
  const lobby = $('#engineLobby'), message = $('#engineLobbyMessage');
  const choice = options.model_choices[choiceKey];
  const hostLimit = Number($(`[data-memory-weight="${choice.weight_tier}"]`).value);
  lobby.classList.add('loading');
  message.textContent = tr(`正在加载${choice.label}（显存自动匹配，H3进程内存上限${hostLimit}GiB），首次进入可能需要几十秒…`);
  message.hidden = false;
  renderEngineLoadProgress({status:'loading', progress_percent:1, progress_stage:'starting', progress_detail:'正在提交模型加载请求'}, true);
  $$('[data-enter-engine]').forEach(button => button.disabled = true);
  try {
    await api('/api/v1/engine', {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({service_family:choice.service_family, weight_tier:choice.weight_tier, host_memory_limit_gib:hostLimit}),
    });
    await reloadOptions();
    switchPage('tasks');
  } catch (error) {
    // The model load may have completed even if the browser lost the long PUT
    // response. Reconcile with server truth before reporting a false failure.
    const recovered = await reconcileEngineState(true).catch(() => false);
    if (!recovered) {
      message.textContent = `进入失败：${error.message}`;
      $$('[data-enter-engine]').forEach(button => button.disabled = false);
    }
  } finally { lobby.classList.remove('loading'); }
}

async function exitEngine() {
  if (!confirm(tr('退出当前模式会释放模型热态。确认退出？'))) return;
  const button = $('#exitEngine'); button.disabled = true; button.textContent = '正在释放…';
  try {
    await api('/api/v1/engine', {method:'DELETE'});
    await reloadOptions();
  } catch (error) { alert(tr(`退出失败：${error.message}`)); }
  finally { button.disabled = false; button.textContent = '切换模型'; }
}

function applyOptions(document) {
  const previousEngine = currentEngine;
  const previousLauncher = currentLauncher;
  options = document;
  currentEngine = options.current_engine;
  currentLauncher = options.current_launcher;
  synchronizeResolutionOptions();
  renderEngineLobby();
  renderReferenceMediaPolicy();
  renderFaceRepairPolicy();
  renderSecondSamplingWindowPolicy();
  if (currentEngine) {
    applyEngineIdentity();
    switchPage(activePage);
  }
  updateSubmitAvailability();
  updateContract();
}

async function reloadOptions() {
  applyOptions(await (await api('/api/v1/options')).json());
  await checkHealth();
}

async function reconcileEngineState(force=false) {
  if (engineReconcilePromise) return engineReconcilePromise;
  engineReconcilePromise = (async () => {
    const document = await (await api('/api/v1/options')).json();
    const lobbyStuck = $('#engineLobby').classList.contains('loading');
    const changed = document.current_engine !== currentEngine
      || document.current_launcher !== currentLauncher
      || Boolean(document.engine_control?.switching) !== Boolean(options?.engine_control?.switching);
    if (force || changed || lobbyStuck) {
      applyOptions(document);
    }
    return Boolean(document.current_engine) && !document.engine_control?.switching;
  })();
  try { return await engineReconcilePromise; }
  finally { engineReconcilePromise = null; }
}

function synchronizeResolutionOptions() {
  const control = $('#creationResolutionControl');
  if (!control || !options) return;
  const field = $('[name="resolution"]', control);
  const slider = $('#creationResolutionSlider');
  const allowedValues = (options.progressive_resolutions || options.resolutions || [])
    .map(value => value === '2k' ? 1440 : Number.parseInt(value, 10))
    .filter(Number.isFinite);
  const policy = options.progressive_resolution || options.resolution || {};
  const minimum = Number(policy.min) || 360;
  const maximum = Math.min(Number(policy.max) || 1080, Math.max(minimum, ...allowedValues));
  slider.min = String(minimum);
  slider.max = String(maximum);
  const firstSlider = $('#firstPassResolutionSlider');
  if (firstSlider) {
    firstSlider.min = String(minimum);
    firstSlider.max = String(maximum);
  }
  $$('[data-resolution]', control).forEach(button => {
    const point = Number.parseInt(button.dataset.resolution, 10);
    button.disabled = point < minimum || point > maximum;
  });
  const current = Number.parseInt(field.value, 10);
  const fallback = Number.parseInt(options.defaults?.resolution, 10) || 480;
  setCreationResolution(Number.isFinite(current) ? current : fallback, true);
}

function canvasForShortEdge(shortEdge, ratio) {
  const [rw, rh] = String(ratio || '16:9').split(':').map(Number);
  const align32 = value => Math.max(32, Math.floor(value / 32 + 0.5) * 32);
  return rw >= rh
    ? {width:align32(shortEdge * rw / rh), height:align32(shortEdge)}
    : {width:align32(shortEdge), height:align32(shortEdge * rh / rw)};
}

function renderCreationResolution() {
  const control = $('#creationResolutionControl');
  if (!control) return;
  const value = $('[name="resolution"]', control).value;
  const shortEdge = Number.parseInt(value, 10);
  const firstEdge = Number.parseInt($('#firstPassResolution')?.value, 10) || shortEdge;
  const english = window.H3I18n?.locale === 'en';
  $('#creationResolutionValue').textContent = firstEdge === shortEdge
    ? `${shortEdge}P · ${english ? 'One resolution throughout' : '全程同分辨率'}`
    : `${english ? 'First' : '一采'} ${firstEdge}P · ${english ? 'Final' : '二采'} ${shortEdge}P`;
  const finalValue = $('#finalPassResolutionValue');
  if (finalValue) finalValue.textContent = `${shortEdge}P`;
  $$('[data-resolution]', control).forEach(button => {
    const selectedPoint = Number.parseInt(button.dataset.resolution, 10) === shortEdge;
    button.classList.toggle('active', selectedPoint);
    button.setAttribute('aria-pressed', String(selectedPoint));
  });
}

function setCreationResolution(raw, forceDetent=false) {
  const control = $('#creationResolutionControl');
  const field = $('[name="resolution"]', control);
  const slider = $('#creationResolutionSlider');
  const minimum = Number(slider.min), maximum = Number(slider.max);
  let value = Math.max(minimum, Math.min(maximum, Math.round(Number.parseFloat(raw))));
  const nearest = CREATION_RESOLUTION_DETENTS.reduce(
    (best, item) => Math.abs(item - value) < Math.abs(best - value) ? item : best,
    CREATION_RESOLUTION_DETENTS[0],
  );
  const hit = nearest >= minimum && nearest <= maximum
    && (forceDetent || Math.abs(nearest - value) <= 10);
  if (hit) value = nearest;
  field.value = `${value}p`;
  slider.value = String(value);
  control.classList.toggle('detent-hit', hit);
  const nativeMaximum = Number(options?.progressive_resolution?.first_pass_max)
    || Number(options?.resolution?.max) || 1080;
  const variant = $('[name="model_variant"]');
  if (value > nativeMaximum && variant?.value !== 'lora') {
    variant.value = 'lora';
    applyEngineIdentity();
  }
  renderCreationResolution();
  syncDurationControl();
  updateSelfLiftControls();
}

function apiHeaders() {
  const headers = {};
  const key = localStorage.getItem('h3serve_api_key');
  if (key) headers['X-API-Key'] = key;
  return headers;
}

async function api(path, init={}) {
  const request = {...init};
  const method = String(request.method || 'GET').toUpperCase();
  const timeoutMs = Number(request.timeoutMs ?? (method === 'GET' ? 10000 : 0));
  delete request.timeoutMs;
  request.headers = {...apiHeaders(), ...(request.headers || {})};
  const controller = timeoutMs > 0 && !request.signal ? new AbortController() : null;
  if (controller) request.signal = controller.signal;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(path, request);
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response;
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error('服务响应超时；请检查8090端口转发，或等待当前计算阶段结束');
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
}


async function serverReferenceMediaSettings() {
  const response = await api('/api/v1/settings/reference-media');
  return response.json();
}

async function configureServerReferenceMedia(imageResolution, videoResolution) {
  const response = await api('/api/v1/settings/reference-media', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      image_resolution:String(imageResolution || '').trim(),
      video_resolution:String(videoResolution || '').trim(),
    }),
  });
  return response.json();
}

async function serverFaceRepairSettings() {
  const response = await api('/api/v1/settings/face-repair');
  return response.json();
}

async function configureServerFaceRepair(canvasSize, capacity) {
  const response = await api('/api/v1/settings/face-repair', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({canvas_size:Number(canvasSize), capacity:Number(capacity)}),
  });
  return response.json();
}

async function serverPreviewSettings() {
  const response = await api('/api/v1/settings/checkpoint-preview');
  return response.json();
}

async function configureServerPreview(steps) {
  const response = await api('/api/v1/settings/checkpoint-preview', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({steps:Number(steps)}),
  });
  return response.json();
}

async function serverSecondSamplingWindowSettings() {
  const response = await api('/api/v1/settings/second-sampling-window');
  return response.json();
}

async function configureServerSecondSamplingWindow(enabled, windowSeconds, overlapSeconds) {
  const response = await api('/api/v1/settings/second-sampling-window', {
    method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      enabled:Boolean(enabled),
      window_seconds:Number(windowSeconds),
      overlap_seconds:Number(overlapSeconds),
    }),
  });
  return response.json();
}

function renderSecondSamplingWindowPolicy(document=null) {
  if (document) {
    globalSecondSamplingWindowPolicy = {
      ...globalSecondSamplingWindowPolicy,
      ...document,
    };
  } else if (options?.second_sampling_window) {
    globalSecondSamplingWindowPolicy = {
      ...globalSecondSamplingWindowPolicy,
      ...options.second_sampling_window,
    };
  }
  const enabled = Boolean(globalSecondSamplingWindowPolicy.enabled);
  const seconds = Math.max(3, Math.min(15,
    Number(globalSecondSamplingWindowPolicy.window_seconds) || 5));
  const overlap = Math.max(0, Math.min(4,
    Number(globalSecondSamplingWindowPolicy.overlap_seconds) || 0));
  globalSecondSamplingWindowPolicy.enabled = enabled;
  globalSecondSamplingWindowPolicy.window_seconds = seconds;
  globalSecondSamplingWindowPolicy.overlap_seconds = overlap;
  const toggle = $('#globalSecondSamplingWindowEnabled');
  const slider = $('#globalSecondSamplingWindowSeconds');
  const output = $('#globalSecondSamplingWindowValue');
  const overlapSlider = $('#globalSecondSamplingOverlapSeconds');
  const overlapOutput = $('#globalSecondSamplingOverlapValue');
  const summary = $('#globalSecondSamplingWindowSummary');
  const controls = $('#globalSecondSamplingWindowControls');
  const badge = $('#globalSecondSamplingWindowBadge');
  if (toggle) toggle.checked = enabled;
  const singleToggle = $('#singleSecondSamplingWindowEnabled');
  if (singleToggle && !singleToggle.dataset.userTouched) {
    singleToggle.checked = enabled;
  }
  if (controls) controls.hidden = !enabled;
  if (badge) badge.textContent = window.H3I18n?.locale === 'en'
    ? (enabled ? 'Enabled' : 'Disabled')
    : (enabled ? '已开启' : '已关闭');
  if (slider) {
    slider.value = String(seconds);
    slider.disabled = !enabled;
  }
  if (overlapSlider) {
    overlapSlider.value = String(overlap);
    overlapSlider.disabled = !enabled;
  }
  if (output) output.textContent = window.H3I18n?.locale === 'en'
    ? `${seconds.toFixed(1)} s` : `${seconds.toFixed(1)} 秒`;
  if (overlapOutput) overlapOutput.textContent = window.H3I18n?.locale === 'en'
    ? `${overlap.toFixed(1)} s` : `${overlap.toFixed(1)} 秒`;
  if (summary) {
    const effective = Number(globalSecondSamplingWindowPolicy.effective_window_seconds);
    const effectiveOverlap = Number(globalSecondSamplingWindowPolicy.effective_overlap_seconds);
    const suffix = Number.isFinite(effective) && Number.isFinite(effectiveOverlap)
      ? (window.H3I18n?.locale === 'en'
          ? ` Effective H3 view: ${effective.toFixed(3)} s; previous-view context: ${effectiveOverlap.toFixed(3)} s.`
          : ` H3 实际对齐窗口：${effective.toFixed(3)} 秒；参考上一窗口：${effectiveOverlap.toFixed(3)} 秒。`)
      : '';
    summary.textContent = window.H3I18n?.locale === 'en'
      ? `${enabled ? 'Enabled' : 'Disabled'} for the high-resolution tail; authored windows and final duration stay unchanged.${suffix}`
      : `${enabled ? '已开启' : '已关闭'}高分辨率尾段滑窗；创作窗口和最终视频时长不变。${suffix}`;
  }
  updateSelfLiftControls();
}

function renderPreviewPolicy(document=null) {
  if (document) globalPreviewPolicy = {...globalPreviewPolicy, ...document};
  const steps = Math.max(1, Math.min(4, Number(globalPreviewPolicy.steps) || 2));
  globalPreviewPolicy.steps = steps;
  const input = $('#globalPreviewSteps');
  const output = $('#globalPreviewStepsValue');
  if (input) input.value = String(steps);
  if (output) output.textContent = window.H3I18n?.locale === 'en'
    ? `${steps} steps` : `${steps} 步`;
  updateSelfLiftControls();
}

async function serverLoraSettings() {
  const response = await api('/api/v1/settings/lora');
  return response.json();
}

async function configureServerLora(checkpoint) {
  const response = await api('/api/v1/settings/lora', {
    method:'PUT', timeoutMs:0,
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({checkpoint:String(checkpoint || '').trim()}),
  });
  return response.json();
}

function renderGlobalLoraPolicy(document=null) {
  if (document) globalLoraPolicy = document;
  const policy = globalLoraPolicy || {};
  const select = $('#globalLoraCheckpoint');
  const badge = $('#globalLoraBadge');
  const status = $('#globalLoraStatus');
  const button = $('#loadGlobalLora');
  if (!select || !badge || !status || !button) return;
  select.replaceChildren();
  const available = Array.isArray(policy.available) ? policy.available : [];
  if (!available.length) {
    const option = new Option(tr('未发现 LoRA 权重'), '');
    option.disabled = true; option.selected = true; select.add(option);
  } else {
    available.forEach(item => {
      const size = Number(item.bytes) > 0 ? ` · ${(Number(item.bytes) / 1073741824).toFixed(2)} GiB` : '';
      const compatibility = item.compatible ? '' : ` · ${tr('格式不兼容')}`;
      const profile = item.profile || {};
      const label = profile.display_name || item.id;
      const families = Array.isArray(profile.task_families)
        ? profile.task_families.map(value => value === 'reference' ? 'Ref2VA' : value === 'first_last' ? 'FL2VA' : value).join('/')
        : '';
      const steps = Array.isArray(profile.recommended_steps) && profile.recommended_steps.length
        ? ` · ${profile.recommended_steps.join('/')}${isEnglish() ? ' steps' : '步'}`
        : '';
      const option = new Option(`${label}${families ? ` · ${families}` : ''}${steps}${size}${compatibility}`, item.id);
      option.disabled = !item.compatible;
      select.add(option);
    });
    if (policy.selected && available.some(item => item.id === policy.selected && item.compatible)) {
      select.value = policy.selected;
    }
  }
  const compatibleCount = available.filter(item => item.compatible).length;
  badge.textContent = policy.changing ? tr('切换中') : `${compatibleCount} ${tr('个可用')}`;
  status.textContent = policy.changing
    ? tr('正在释放并重建当前 H3 热引擎，请勿提交任务。')
    : policy.selected
      ? `${tr('当前版本')}：${policy.selected}${policy.loaded ? ` · ${tr('热引擎已加载')}：${policy.loaded}` : ''}`
      : tr('选择一个兼容权重；进入模型后加载会重建热引擎。');
  select.disabled = Boolean(policy.changing) || compatibleCount === 0;
  button.disabled = select.disabled || !select.value;
}

async function loadSelectedGlobalLora() {
  const button = $('#loadGlobalLora');
  const checkpoint = $('#globalLoraCheckpoint').value;
  if (!checkpoint) return;
  button.disabled = true;
  button.textContent = tr('正在重建引擎…');
  $('#globalLoraStatus').textContent = tr('切换要求队列为空；当前热引擎将完整释放并重新加载。');
  try {
    const result = await configureServerLora(checkpoint);
    renderGlobalLoraPolicy(result);
    $('#globalLoraStatus').textContent = result.loaded
      ? `${tr('加载完成')}：${result.loaded}`
      : `${tr('已设为待加载版本')}：${result.selected}`;
    await reloadOptions();
  } catch (error) {
    $('#globalLoraStatus').textContent = `${tr('加载失败')}：${error.message}`;
  } finally {
    button.textContent = tr('加载所选 LoRA');
    button.disabled = !$('#globalLoraCheckpoint').value;
  }
}

function resolutionPolicyLabel(value) {
  return value === 'original'
    ? (isEnglish() ? 'original resolution' : '原分辨率')
    : String(value || '').toUpperCase();
}

function renderReferenceMediaPolicy(document=null) {
  const policy = document || options?.reference_media_processing || {};
  const image = policy.image_resolution || policy.image_default || '720p';
  const video = policy.video_resolution || policy.video_default || '360p';
  const imageControl = $('#globalReferenceImageResolution');
  const videoControl = $('#globalReferenceVideoResolution');
  if (imageControl) imageControl.value = image;
  if (videoControl) videoControl.value = video;
  const hint = $('#referenceMediaPolicyHint');
  if (hint) {
    hint.textContent = isEnglish()
      ? `Automatic proportional downsampling: images ${resolutionPolicyLabel(image)} · videos ${resolutionPolicyLabel(video)}`
      : `自动等比降分辨率：图片${resolutionPolicyLabel(image)} · 视频${resolutionPolicyLabel(video)}`;
  }
}

function renderFaceRepairPolicy(document=null) {
  const policy = document || options?.face_repair || globalFaceRepairPolicy;
  globalFaceRepairPolicy = {
    canvas_size:Math.max(192, Math.min(1088, Number(policy.canvas_size) || 768)),
    capacity:[1,4,9,16].includes(Number(policy.capacity)) ? Number(policy.capacity) : 4,
  };
  const canvas = $('#globalFaceRepairCanvas');
  const capacity = $('#globalFaceRepairCapacity');
  if (canvas) canvas.value = String(globalFaceRepairPolicy.canvas_size);
  if (capacity) capacity.value = String(globalFaceRepairPolicy.capacity);
  updateFaceRepairSettingsOutputs();
}

function updateFaceRepairSettingsOutputs() {
  const canvas = Number($('#globalFaceRepairCanvas')?.value ?? globalFaceRepairPolicy.canvas_size);
  const canvasOutput = $('#globalFaceRepairCanvasValue');
  if (canvasOutput) canvasOutput.textContent = `${canvas}P`;
}

function selected(name) { return $(`[name="${name}"]`).value; }
function escapeHtml(value='') { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
function isEnglish() { return window.H3I18n?.locale === 'en'; }
function formatSeconds(value) {
  if (value == null || !Number.isFinite(Number(value))) return isEnglish() ? 'Estimating' : '估算中';
  const seconds = Math.max(0, Math.round(Number(value)));
  if (seconds < 60) return isEnglish() ? `About ${seconds}s` : `约 ${seconds} 秒`;
  return isEnglish()
    ? `About ${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`
    : `约 ${Math.floor(seconds / 60)}分${String(seconds % 60).padStart(2, '0')}秒`;
}

function formatElapsed(value) {
  if (value == null || !Number.isFinite(Number(value))) return isEnglish() ? 'Not recorded' : '未记录';
  const seconds = Math.max(0, Number(value));
  if (seconds < 60) return isEnglish()
    ? `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`
    : `${seconds.toFixed(seconds < 10 ? 2 : 1)} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return isEnglish()
    ? `${minutes}m ${remainder.toFixed(1).padStart(4, '0')}s`
    : `${minutes}分${remainder.toFixed(1).padStart(4, '0')}秒`;
}

function currentMaxDuration() {
  // The ordinary creation desk represents one native H3 generation task.
  // Longer tail-editable work belongs to Infinite Creation.
  return 15;
}

function framesForDuration(seconds) {
  return 5 + 17 * Math.max(0, Math.round((Number(seconds) * 24 - 5) / 17));
}

function referenceBindingError() {
  if (currentEngine !== 'reference') return '';
  const counts = {
    picture: referenceFiles('image').length,
    video: referenceFiles('video').length,
    audio: referenceFiles('audio').length,
  };
  if (!counts.picture && !counts.video && !counts.audio) {
    return isEnglish()
      ? 'No reference media is currently uploaded. Browsers do not retain local files after a refresh; add the images, videos, or audio again.'
      : '当前没有实际上传的参考素材。刷新页面后浏览器不会保留本地文件，请重新添加图片、视频或音频。';
  }
  const labels = ($('#freeformPrompt')?.value || '').matchAll(/<(Picture|Video|Audio)\s+(\d+)>/gi);
  for (const match of labels) {
    const kind = match[1].toLowerCase();
    const index = Number(match[2]);
    if (index < 1 || index > counts[kind]) {
      return isEnglish()
        ? `<${match[1]} ${index}> has no matching uploaded file. Add the media again or remove this reference.`
        : `<${match[1]} ${index}> 没有对应的已上传文件，请重新添加素材或删除这个引用。`;
    }
  }
  return '';
}

function referenceFiles(kind) {
  if (kind === 'image') return Array.from($('#referenceImages')?.files || []).slice(0, 9);
  if (kind === 'video') return Array.from($('#referenceVideos')?.files || []).slice(0, 3);
  return Array.from($('#referenceAudios')?.files || []).slice(0, 3);
}

function renderReferencePreviews(kind) {
  const files = referenceFiles(kind);
  const preview = kind === 'image' ? $('#referencePreview') : kind === 'video' ? $('#referenceVideoPreview') : $('#referenceAudioPreview');
  preview.innerHTML = files.map((file, index) => {
    const label = kind === 'image' ? `Picture ${index + 1}` : kind === 'video' ? `Video ${index + 1}` : `Audio ${index + 1}`;
    const visual = kind === 'image'
      ? `<img src="${URL.createObjectURL(file)}" alt="${isEnglish() ? 'Reference image' : '参考图片'} ${index + 1}">`
      : kind === 'video'
        ? `<video src="${URL.createObjectURL(file)}" muted preload="metadata"></video>`
        : `<span class="reference-audio-icon">♫</span>`;
    return `<article class="reference-chip" data-reference-kind="${kind}" data-reference-index="${index}"><button type="button" class="reference-chip-remove" data-reference-remove aria-label="删除素材">×</button>
      <div class="reference-media-visual">${visual}</div>
      <div class="reference-chip-copy"><button type="button" class="reference-token" data-insert-reference title="插入当前提示词">&lt;${label}&gt;</button><small title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</small></div>
    </article>`;
  }).join('');
  $$('[data-reference-remove]', preview).forEach((button, index) => button.addEventListener('click', () => removeReferenceFile(kind, index)));
  $$('[data-insert-reference]', preview).forEach((button, index) => button.addEventListener('click', () => insertReferenceToken(kind, index)));
}

function insertReferenceToken(kind, index) {
  const prefix = kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio';
  const token = `<${prefix} ${index + 1}>`;
  insertTokenIntoTextarea($('#freeformPrompt'), token, false);
}

function insertTokenIntoTextarea(target, token, replaceMention=false) {
  const start = target.selectionStart ?? target.value.length;
  const end = target.selectionEnd ?? start;
  let before = target.value.slice(0, start);
  if (replaceMention && before.endsWith('@')) before = before.slice(0, -1);
  const spacer = before && !/\s$/.test(before) ? ' ' : '';
  target.value = `${before}${spacer}${token} ${target.value.slice(end)}`;
  target.dispatchEvent(new Event('input', {bubbles:true}));
  const cursor = before.length + spacer.length + token.length + 1;
  target.focus(); target.setSelectionRange(cursor, cursor);
}

function referenceMentionItems() {
  return ['image','video','audio'].flatMap(kind => referenceFiles(kind).map((file, index) => ({
    kind, index, file, token:`<${kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio'} ${index + 1}>`,
  })));
}

function updateReferenceMentionMenu(card, textarea) {
  const menu = $('.reference-mention-menu', card);
  if (currentEngine !== 'reference') { menu.hidden = true; return; }
  const before = textarea.value.slice(0, textarea.selectionStart ?? 0);
  if (!before.endsWith('@')) { menu.hidden = true; return; }
  const items = referenceMentionItems();
  menu.innerHTML = items.length ? items.map(item => `<button type="button" data-mention-kind="${item.kind}" data-mention-index="${item.index}"><b>${escapeHtml(item.token)}</b><small>${escapeHtml(item.file.name)}</small></button>`).join('') : '<span>请先添加参考文件</span>';
  menu.hidden = false;
  $$('button', menu).forEach(button => button.addEventListener('mousedown', event => {
    event.preventDefault();
    const kind = button.dataset.mentionKind;
    const index = Number(button.dataset.mentionIndex);
    const token = `<${kind === 'image' ? 'Picture' : kind === 'video' ? 'Video' : 'Audio'} ${index + 1}>`;
    insertTokenIntoTextarea(textarea, token, true);
    menu.hidden = true;
  }));
}

function removeReferenceFile(kind, removeIndex) {
  const input = kind === 'image' ? $('#referenceImages') : kind === 'video' ? $('#referenceVideos') : $('#referenceAudios');
  const transfer = new DataTransfer();
  Array.from(input.files || []).forEach((file, index) => { if (index !== removeIndex) transfer.items.add(file); });
  input.files = transfer.files;
  renderReferencePreviews(kind);
  updateContract();
}

function syncDurationControl() {
  const input = $('[name="duration_seconds"]');
  const maximum = currentMaxDuration();
  input.max = String(maximum);
  input.value = String(Math.max(1, Math.min(maximum, Number(input.value) || 5)));
  $('#freeformDurationValue').textContent = isEnglish()
    ? `${Number(input.value).toFixed(1)}s`
    : `${Number(input.value).toFixed(1)} 秒`;
  updateContract();
}

function switchPage(page) {
  if (!currentEngine) return;
  if (!['create','infinite','tasks'].includes(page)) page = 'tasks';
  activePage = page;
  $$('.workspace-tabs button').forEach(button => button.classList.toggle('active', button.dataset.page === page));
  $('#createPage').hidden = page !== 'create';
  $('#infinitePage').hidden = page !== 'infinite';
  $('#tasksPage').hidden = page !== 'tasks';
  $('#createPage').classList.toggle('active', page === 'create');
  $('#infinitePage').classList.toggle('active', page === 'infinite');
  $('#tasksPage').classList.toggle('active', page === 'tasks');
  if (page === 'tasks') refreshJobs();
  if (page === 'infinite') window.H3InfiniteStudio?.activate(currentEngine, options);
}

function currentGeometry() {
  const resolution = selected('resolution');
  const ratio = selected('aspect_ratio');
  return options.geometry?.[resolution]?.[ratio]
    || canvasForShortEdge(Number.parseInt(resolution, 10), ratio);
}

function updateSettingsSummaries() {
  const english = isEnglish();
  const inference = $('#inferenceSummary');
  if (inference) {
    const baseTier = options?.active_weight_tier === 'w4a8' ? 'W4A8' : 'INT8';
    const variant = currentVariant() === 'lora' ? `${baseTier}+LoRA` : baseTier;
    const steps = Number(selected('sampling_steps')) || (currentVariant() === 'lora' ? 8 : 20);
    const firstSteps = Math.max(1, Math.min(steps, Number($('#firstPassSteps')?.value) || steps));
    const acceleration = Number(selected('acceleration')) || 0;
    const secondAcceleration = Number(selected('second_pass_acceleration')) || 0;
    const targetResolution = String(selected('resolution')).toUpperCase();
    const firstResolution = String($('#firstPassResolution')?.value || selected('resolution')).toUpperCase();
    const accelerationLabel = value => value ? `${english ? 'acceleration' : '加速'} ${value}` : 'Dense';
    const resolutionPlan = firstResolution === targetResolution
      ? `${targetResolution} ${english ? 'throughout' : '全程'}`
      : `${firstResolution}→${targetResolution}`;
    const preview = $('#previewEnabled')?.checked
      ? ` · ${english ? 'intermediate preview' : '中途预览'}`
      : '';
    const accelerationPlan = firstSteps < steps
      ? english
        ? `first ${accelerationLabel(acceleration)} / final ${accelerationLabel(secondAcceleration)}`
        : `一采${accelerationLabel(acceleration)} / 二采${accelerationLabel(secondAcceleration)}`
      : accelerationLabel(acceleration);
    inference.textContent = english
      ? `${variant} · first ${firstSteps}/${steps} total steps · ${accelerationPlan} · ${resolutionPlan}${preview}`
      : `${variant} · 一采${firstSteps}/总${steps}步 · ${accelerationPlan} · ${resolutionPlan}${preview}`;
  }
  const video = $('#videoSettingsSummary');
  if (video && options) {
    const geometry = currentGeometry();
    const source = `${String(selected('resolution')).toUpperCase()} · ${selected('aspect_ratio')} · ${geometry.width}×${geometry.height}`;
    video.textContent = `${source} · ${Number(selected('duration_seconds')).toFixed(1)}${english ? 's' : '秒'}`;
  }
}

function updateJointAccelerationControls() {
  if (!options) return;
  const variant = currentVariant();
  syncModelVariantToggle();
  const limits = options.advanced_limits?.sampling_steps?.[variant] || (variant === 'lora'
    ? {min:4, max:10, default:8}
    : {min:5, max:30, default:20});
  const steps = $('[name="sampling_steps"]');
  const firstSteps = $('#firstPassSteps');
  const acceleration = $('[name="acceleration"]');
  const secondAcceleration = $('[name="second_pass_acceleration"]');
  const available = Boolean(options.advanced_limits?.sparse_attention_available);
  const accelerationLimits = options.advanced_limits?.acceleration || {min:0, max:100, step:1};
  steps.min = '1';
  steps.max = String(limits.max);
  firstSteps.min = '1';
  firstSteps.max = String(limits.max);
  const selectedLoRA = (globalLoraPolicy?.available || []).find(
    item => item.id === globalLoraPolicy?.selected
  );
  const loraProfile = selectedLoRA?.profile || {};
  const loraProfileId = variant === 'lora' ? String(loraProfile.profile_id || '') : '';
  const profileDefault = variant === 'lora'
    ? Number(loraProfile.default_steps) || limits.default
    : limits.default;
  const trajectoryChanged = (
    steps.dataset.variant !== variant
    || steps.dataset.loraProfile !== loraProfileId
  );
  if (trajectoryChanged) {
    steps.value = String(profileDefault);
    steps.dataset.variant = variant;
    steps.dataset.loraProfile = loraProfileId;
    firstSteps.value = String(Math.max(1, profileDefault - 2));
  }
  steps.value = String(Math.max(limits.min, Math.min(limits.max, Number(steps.value) || profileDefault)));
  steps.disabled = false;
  const recommendedSteps = Array.isArray(loraProfile.recommended_steps)
    ? loraProfile.recommended_steps.map(Number)
    : [];
  steps.dataset.modelHint = variant === 'lora'
    ? recommendedSteps.length && !recommendedSteps.includes(Number(steps.value))
      ? isEnglish()
        ? `${loraProfile.display_name || 'LoRA'} recommends ${recommendedSteps.join('/')} steps. Other values can run but are outside the distillation calibration points.`
        : `当前 ${loraProfile.display_name || 'LoRA'} 建议 ${recommendedSteps.join('/')} 步；其他步数可运行，但不在蒸馏标定点。`
      : tr(Number(steps.value) > 8
        ? '超过8步未经LoRA质量校准；允许运行，但不保证质量随步数单调增加。'
        : 'LoRA 使用完整 Turbo 步；内部加速只分配逐步逐层注意力，不擅自加入预测步。')
    : tr('决定完整 σ 去噪轨迹长度；系统在这条轨迹内联合安排真实步和预测步。');
  if (!available) {
    for (const control of [acceleration, secondAcceleration]) {
      control.value = '0';
      control.min = '0';
      control.max = '0';
      control.disabled = true;
    }
    $('#accelerationAdvanced').classList.add('unavailable');
    $('#secondPassAccelerationAdvanced').classList.add('unavailable');
    $('#accelerationSafety').textContent = tr('当前服务未安装 SM89 稀疏运行时，只能使用 0（Dense）；请运行项目安装脚本。');
    $('#secondPassAccelerationSafety').textContent = tr('当前服务未安装 SM89 稀疏运行时，只能使用 0（Dense）；请运行项目安装脚本。');
  } else {
    for (const control of [acceleration, secondAcceleration]) {
      control.min = String(accelerationLimits.min);
      control.max = String(accelerationLimits.max);
      control.step = String(accelerationLimits.step);
      control.disabled = false;
    }
    $('#accelerationAdvanced').classList.remove('unavailable');
    $('#secondPassAccelerationAdvanced').classList.remove('unavailable');
    const scheduler = accelerationLimits.scheduler_by_variant?.[variant]
      || accelerationLimits.scheduler;
    const certified = scheduler === 'v19_certified_frontier';
    const paretoV24 = String(scheduler || '').startsWith('h3_pareto_v24');
    const qualityKnee = Number(accelerationLimits.quality_knee) || 75;
    $('#accelerationSafety').textContent = tr(variant === 'lora'
      ? 'LoRA 无预测调度：全部 Turbo 步保持真实计算，档位只改变逐步逐层 Attention 配额。'
      : paretoV24
        ? `V24统一帕累托调度：0为Dense，${qualityKnee}为Human审核的发布质量拐点，${qualityKnee}–100为允许肉眼缺陷的激进区。`
      : certified
        ? 'V19认证前沿：仅命中已封存工作负载时加速；其他输入自动Dense回退。'
        : '冻结的 Round229 调度：构图锚点、因果层、预测后恢复步和末端细节保护始终开启。');
    $('#secondPassAccelerationSafety').textContent = tr('控制绿色分叉点到橙色终点的正式高清分支。');
  }
  const level = Math.max(0, Math.min(100, Number(acceleration.value) || 0));
  const secondLevel = Math.max(0, Math.min(100, Number(secondAcceleration.value) || 0));
  const qualityKnee = Number(accelerationLimits.quality_knee) || 75;
  const activeScheduler = accelerationLimits.scheduler_by_variant?.[variant]
    || accelerationLimits.scheduler;
  const hasHumanKnee = variant === 'base'
    && String(activeScheduler || '').startsWith('h3_pareto_v24');
  $('#accelerationValue').textContent = level === 0
    ? '0 · Dense'
    : hasHumanKnee && level === qualityKnee
      ? `${level} · ${isEnglish() ? 'release quality knee' : '发布质量拐点'}`
      : hasHumanKnee && level === 100
        ? `100 · ${isEnglish() ? 'aggressive' : '激进'}`
        : `${level} / 100`;
  $('#secondPassAccelerationValue').textContent = secondLevel === 0
    ? '0 · Dense'
    : hasHumanKnee && secondLevel === qualityKnee
      ? `${secondLevel} · ${isEnglish() ? 'release quality knee' : '发布质量拐点'}`
      : hasHumanKnee && secondLevel === 100
        ? `100 · ${isEnglish() ? 'aggressive' : '激进'}`
        : `${secondLevel} / 100`;
  updateSelfLiftControls();
}

function updateContract() {
  if (!options || !currentEngine) return;
  const geometry = currentGeometry();
  const width = geometry.width;
  const height = geometry.height;
  const requested = Number(selected('duration_seconds')) || 5;
  const frames = framesForDuration(requested);
  const geometryText = $('#geometryText');
  const durationText = $('#durationText');
  if (geometryText) geometryText.textContent = `${width} × ${height}`;
  const english = isEnglish();
  if (durationText) durationText.textContent = english
    ? `${(frames / 24).toFixed(2)}s · ${frames} frames`
    : `${(frames / 24).toFixed(2)}秒 · ${frames}帧`;
  const first = $('[name="first_frame"]').files.length > 0;
  const last = $('[name="last_frame"]').files.length > 0;
  const references = $('#referenceImages')?.files?.length || 0;
  const referenceVideos = $('#referenceVideos')?.files?.length || 0;
  const referenceAudios = $('#referenceAudios')?.files?.length || 0;
  const conditionText = $('#conditionText');
  if (conditionText) conditionText.textContent = english
    ? currentEngine === 'reference'
      ? `Multi-reference video · ${references} images / ${referenceVideos} videos / ${referenceAudios} audio files`
      : first && last ? 'First/last-frame-to-video' : first ? 'First-frame-to-video' : last ? 'Last-frame-to-video' : 'Text-to-video'
    : currentEngine === 'reference'
      ? `多参考生视频 · ${references}图 / ${referenceVideos}视频 / ${referenceAudios}音频`
      : first && last ? '首尾帧生视频' : first ? '首帧生视频' : last ? '尾帧生视频' : '文生视频';
  updateSelfLiftControls();
  updateSettingsSummaries();
}

function solverStepCount() {
  return Number(selected('sampling_steps')) || (currentVariant() === 'lora' ? 8 : 20);
}

function updateSelfLiftControls() {
  const control = $('#creationResolutionControl');
  const slider = $('#firstPassResolutionSlider');
  const targetSlider = $('#creationResolutionSlider');
  const field = $('#firstPassResolution');
  const firstSteps = $('#firstPassSteps');
  const totalSteps = $('[name="sampling_steps"]');
  if (!control || !slider || !targetSlider || !field || !firstSteps || !totalSteps) return;
  const target = Number.parseInt(selected('resolution'), 10) || 480;
  const minimum = Number(options?.resolution?.min) || 360;
  const nativeMaximum = Math.min(
    target,
    Number(options?.progressive_resolution?.first_pass_max)
      || Number(options?.resolution?.max) || 1080,
  );
  const progressiveAvailable = target > minimum;
  slider.min = targetSlider.min;
  slider.max = targetSlider.max;
  slider.disabled = false;
  control.classList.remove('first-pass-unavailable');
  let initial = Number.parseInt(field.value, 10);
  if (!Number.isFinite(initial)) initial = target;
  initial = progressiveAvailable
    ? Math.max(minimum, Math.min(nativeMaximum, initial))
    : Math.min(target, nativeMaximum);
  slider.value = String(initial);
  field.value = `${initial}p`;
  const progressive = progressiveAvailable && initial < target;
  const english = window.H3I18n?.locale === 'en';
  const rangeMinimum = Number(targetSlider.min) || minimum;
  const rangeMaximum = Math.max(rangeMinimum + 1, Number(targetSlider.max) || target);
  const resolutionPercent = value => (
    Math.max(0, Math.min(1, (value - rangeMinimum) / (rangeMaximum - rangeMinimum))) * 100
  );
  const resolutionTrack = $('#creationResolutionTrack');
  resolutionTrack.style.setProperty('--first-step-percent', `${resolutionPercent(initial)}%`);
  resolutionTrack.style.setProperty('--total-step-percent', `${resolutionPercent(target)}%`);
  const total = Math.max(1, Number(totalSteps.value) || solverStepCount());
  const maximumFirst = progressive ? Math.max(1, total - 1) : total;
  const first = Math.max(
    1,
    Math.min(maximumFirst, Number(firstSteps.value) || Math.max(1, total - 2)),
  );
  firstSteps.value = String(first);
  firstSteps.max = totalSteps.max;
  const rangeMin = Number(totalSteps.min) || 1;
  const rangeMax = Math.max(rangeMin + 1, Number(totalSteps.max) || total);
  const percentage = value => (
    Math.max(0, Math.min(1, (value - rangeMin) / (rangeMax - rangeMin))) * 100
  );
  const stepTrack = $('#generationStepTrack');
  stepTrack.style.setProperty('--first-step-percent', `${percentage(first)}%`);
  stepTrack.style.setProperty('--total-step-percent', `${percentage(total)}%`);
  $('#firstPassStepsValue').textContent = english ? `${first} steps` : `${first} 步`;
  $('#totalStepsValue').textContent = english ? `${total} steps` : `${total} 步`;
  $('#samplingStepsValue').textContent = english
    ? `First ${first} · ${total} total steps`
    : `一采 ${first} · 总计 ${total} 步`;
  const highResolutionSteps = total - first;
  const previewEnabled = $('#previewEnabled').checked;
  const singleWindowToggle = $('#singleSecondSamplingWindowEnabled');
  const singleWindowControl = $('#singleSecondSamplingWindowControl');
  const singleSigmaField = $('#singleSecondSamplingSigmaField');
  const singleSigma = $('#singleSecondSamplingSigmaScale');
  const singleWindowEnabled = Boolean(
    progressive && singleWindowToggle?.checked
  );
  if (singleWindowToggle) singleWindowToggle.disabled = !progressive;
  if (singleWindowControl) singleWindowControl.classList.toggle('unavailable', !progressive);
  if (singleSigmaField) singleSigmaField.hidden = !singleWindowEnabled;
  if (singleSigma) singleSigma.disabled = !singleWindowEnabled;
  const sigmaScale = Math.max(0.25, Math.min(1, Number(singleSigma?.value) || 1));
  if ($('#singleSecondSamplingSigmaValue')) {
    $('#singleSecondSamplingSigmaValue').textContent = sigmaScale.toFixed(2);
  }
  if ($('#singleSecondSamplingWindowSummary')) {
    $('#singleSecondSamplingWindowSummary').textContent = singleWindowEnabled
      ? `${Number(globalSecondSamplingWindowPolicy.window_seconds).toFixed(1)}s / ${Number(globalSecondSamplingWindowPolicy.overlap_seconds).toFixed(1)}s`
      : (english ? 'Off' : '关闭');
  }
  const secondAcceleration = $('[name="second_pass_acceleration"]');
  const sparseAvailable = Boolean(options?.advanced_limits?.sparse_attention_available);
  const hasFormalTail = first < total;
  secondAcceleration.disabled = !sparseAvailable || !hasFormalTail;
  $('#secondPassAccelerationAdvanced').classList.toggle(
    'unavailable', !sparseAvailable || !hasFormalTail,
  );
  if (sparseAvailable) {
    $('#secondPassAccelerationSafety').textContent = tr(hasFormalTail
      ? '控制绿色分叉点到橙色终点的正式高清分支。'
      : '绿色节点已与橙色终点重合，当前没有二采区间。');
  }
  const previewSteps = Math.max(1, Math.min(4, Number(globalPreviewPolicy.steps) || 2));
  $('#samplingStepsHint').textContent = english
    ? progressive
      ? `Complete ${first} steps at ${initial}P, then ${highResolutionSteps} steps at ${target}P. ${totalSteps.dataset.modelHint || ''}`
      : `Generate at ${target}P throughout; the green handle is at step ${first}. ${totalSteps.dataset.modelHint || ''}`
    : progressive
      ? `${initial}P 完成 ${first} 步，再用 ${target}P 完成 ${highResolutionSteps} 步。${totalSteps.dataset.modelHint || ''}`
      : `${target}P 全程生成；绿色节点位于第 ${first} 步。${totalSteps.dataset.modelHint || ''}`;
  $('#firstPassResolutionValue').textContent = `${initial}P`;
  $('#finalPassResolutionValue').textContent = `${target}P`;
  $('#creationResolutionValue').textContent = progressive
    ? `${english ? 'First' : '一采'} ${initial}P · ${english ? 'Final' : '二采'} ${target}P`
    : `${target}P · ${english ? 'One resolution throughout' : '全程同分辨率'}`;
  $('#firstPassResolutionHint').textContent = progressiveAvailable
    ? progressive
      ? english
        ? `After the green handle, lift to ${target}P. The first-pass resolution never exceeds the final resolution.`
        : `绿色节点完成后放大到 ${target}P；一次采样分辨率始终不超过二次采样分辨率。`
      : tr('两个节点重合时全程使用同一分辨率，不执行 latent 放大。')
    : tr('当前已是最低可选分辨率，全程使用同一分辨率。');
  $('#previewPolicySummary').textContent = previewEnabled
    ? `${english ? 'On' : '开启'} · ${previewSteps} ${english ? 'steps' : '步'}`
    : `${english ? 'Off' : '关闭'} · 0 ${english ? 'steps' : '步'}`;
  updateSettingsSummaries();
}

function setFirstPassResolution(raw, forceDetent=false) {
  const control = $('#creationResolutionControl');
  const slider = $('#firstPassResolutionSlider');
  const field = $('#firstPassResolution');
  if (!control || !slider || !field) return;
  const minimum = Number(slider.min) || 360;
  const target = Number.parseInt(selected('resolution'), 10) || minimum;
  const nativeMaximum = Number(options?.progressive_resolution?.first_pass_max)
    || Number(options?.resolution?.max) || 1080;
  const maximum = Math.min(target, nativeMaximum);
  let value = Math.max(minimum, Math.min(maximum, Math.round(Number.parseFloat(raw))));
  const nearest = FIRST_PASS_RESOLUTION_DETENTS.reduce(
    (best, item) => Math.abs(item - value) < Math.abs(best - value) ? item : best,
    FIRST_PASS_RESOLUTION_DETENTS[0],
  );
  const hit = nearest >= minimum && nearest <= maximum
    && (forceDetent || Math.abs(nearest - value) <= 10);
  if (hit) value = nearest;
  if (Math.abs(maximum - value) <= 10) value = maximum;
  field.value = `${value}p`;
  slider.value = String(value);
  control.classList.toggle('detent-hit', hit || value === maximum);
  updateSelfLiftControls();
}

function applyEngineIdentity() {
  currentEngine = options.current_engine;
  const info = options.current_engine_options;
  if (!currentEngine || !info) return;
  const reference = currentEngine === 'reference';
  const lora = currentVariant() === 'lora';
  const w4a8 = options.active_weight_tier === 'w4a8';
  const vramProfile = String(options.active_vram_profile || (w4a8 ? '8gb' : '24gb')).toUpperCase();
  $('#engineBanner').dataset.engine = currentEngine;
  $('#engineIcon').textContent = reference ? 'R' : 'F';
  $('#engineIcon').className = `engine-icon ${lora ? 'turbo' : 'original'}`;
  $('#engineName').textContent = info.label;
  $('#engineBadge').textContent = `${w4a8 ? 'W4A8' : 'INT8'} · ${vramProfile}${lora ? ' · LoRA' : ''}`;
  const english = isEnglish();
  const backendLabel = english
    ? `${vramProfile} ${w4a8 ? 'low-bit' : 'dedicated high-speed'} backend`
    : `${vramProfile}${w4a8 ? '低比特' : '独立高速'}后端`;
  const activeLoRA = (globalLoraPolicy?.available || []).find(
    item => item.id === globalLoraPolicy?.selected
  );
  const loraLabel = activeLoRA?.profile?.display_name || 'LoRA Turbo';
  $('#engineDescription').textContent = english
    ? reference
      ? `Ref2VA multi-reference · ${backendLabel} · ${lora ? loraLabel : 'base sampling'}`
      : `FL2VA / text-to-video · ${backendLabel} · ${lora ? loraLabel : 'base sampling'}`
    : reference
      ? `Ref2VA 多参考 · ${backendLabel} · ${lora ? loraLabel : '原始采样'}`
      : `FL2VA / 文生视频 · ${backendLabel} · ${lora ? loraLabel : '原始采样'}`;
  $('#brandSubtitle').textContent = reference ? 'Native reference generation' : 'Native first/last generation';
  $('#keyframeFieldset').hidden = reference;
  $('#referenceFieldset').hidden = !reference;
  renderReferenceMediaPolicy();
  if (!reference) $$('.reference-mention-menu').forEach(menu => { menu.hidden = true; });
  updateJointAccelerationControls();
}

function bindDropzone(id) {
  const zone = $(id), input = $('input', zone), image = $('img', zone);
  input.addEventListener('change', () => {
    if (!input.files[0]) return;
    image.src = URL.createObjectURL(input.files[0]); zone.classList.add('has-image'); updateContract();
  });
  $('.remove-frame', zone).addEventListener('click', event => {
    event.preventDefault(); event.stopPropagation(); input.value = ''; image.removeAttribute('src'); zone.classList.remove('has-image'); updateContract();
  });
  bindFileDrop(zone, input, {multiple:false, accept:file => file.type.startsWith('image/')});
}

function bindFileDrop(zone, input, {multiple=true, maxFiles=Infinity, accept=()=>true}={}) {
  const activate = event => { event.preventDefault(); event.stopPropagation(); zone.classList.add('drag-active'); };
  const deactivate = event => { event.preventDefault(); event.stopPropagation(); zone.classList.remove('drag-active'); };
  zone.addEventListener('dragenter', activate);
  zone.addEventListener('dragover', activate);
  zone.addEventListener('dragleave', deactivate);
  zone.addEventListener('drop', event => {
    deactivate(event);
    const files = Array.from(event.dataTransfer?.files || []).filter(accept);
    if (!files.length) return;
    const transfer = new DataTransfer();
    const existing = multiple ? Array.from(input.files || []) : [];
    const identities = new Set(existing.map(file => `${file.name}:${file.size}:${file.lastModified}`));
    const appended = files.filter(file => {
      const identity = `${file.name}:${file.size}:${file.lastModified}`;
      if (identities.has(identity)) return false;
      identities.add(identity); return true;
    });
    (multiple ? [...existing, ...appended].slice(0, maxFiles) : files.slice(0, 1)).forEach(file => transfer.items.add(file));
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', {bubbles:true}));
  });
}

function appendFiles(input, files, maxFiles) {
  const transfer = new DataTransfer();
  const existing = Array.from(input.files || []);
  const identities = new Set(existing.map(file => `${file.name}:${file.size}:${file.lastModified}`));
  const appended = Array.from(files || []).filter(file => {
    const identity = `${file.name}:${file.size}:${file.lastModified}`;
    if (identities.has(identity)) return false;
    identities.add(identity); return true;
  });
  [...existing, ...appended].slice(0, maxFiles).forEach(file => transfer.items.add(file));
  input.files = transfer.files;
  input.dispatchEvent(new Event('change', {bubbles:true}));
}

function distributeReferenceFiles(files) {
  const all = Array.from(files || []);
  appendFiles($('#referenceImages'), all.filter(file => file.type.startsWith('image/')), 9);
  appendFiles($('#referenceVideos'), all.filter(file => file.type.startsWith('video/') || /\.(mp4|mov|mkv|webm|avi)$/i.test(file.name)), 3);
  appendFiles($('#referenceAudios'), all.filter(file => file.type.startsWith('audio/') || /\.(wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name)), 3);
  $('#referenceFiles').value = '';
}

function statusName(job) {
  if (isEnglish()) {
    if (job.progress?.stage === 'cancelling') return 'Cancelling';
    return {
      queued:`Waiting ${job.queue_position || ''}`,
      starting_backend:'Preparing model', running:'Generating', checkpointed:'Checkpoint saved',
      awaiting_preview:'Awaiting preview decision', succeeded:'Completed', failed:'Failed', cancelled:'Cancelled',
    }[job.status] || job.status;
  }
  if (job.progress?.stage === 'cancelling') return '正在取消';
  return {queued:`等待 ${job.queue_position || ''}`,starting_backend:'准备模型',running:'生成中',checkpointed:'断点已保存',awaiting_preview:'等待抽卡决定',succeeded:'已完成',failed:'失败',cancelled:'已取消'}[job.status] || job.status;
}

function progressMarkup(job) {
  const progress = job.progress || {};
  const percent = job.status === 'succeeded' ? 100 : Math.max(0, Math.min(100, Number(progress.percent) || 0));
  if (!['starting_backend','running','queued','awaiting_preview'].includes(job.status)) return '';
  const etaValue = job.status === 'queued' ? progress.estimated_completion_seconds : progress.estimated_remaining_seconds;
  const eta = formatSeconds(etaValue);
  const etaLabel = isEnglish()
    ? job.status === 'queued' ? 'Estimated completion (including queue)' : job.status === 'starting_backend' ? 'Estimated generation after model load' : 'Estimated remaining'
    : job.status === 'queued' ? '预计完成（含排队）' : job.status === 'starting_backend' ? '模型就绪后预计生成' : '预计剩余';
  return `<div class="job-progress"><div class="progress-copy"><span>${escapeHtml(progress.detail || statusName(job))}</span><b>${percent.toFixed(0)}%</b></div><div class="progress-track"><i style="width:${percent}%"></i></div><small>${etaLabel} ${eta}</small></div>`;
}

function memoryExecutionSummary(req, job) {
  const receipt = job?.inference_plan?.memory_execution;
  const labels = isEnglish()
    ? {exact_streaming:'exact streaming', compact_streaming:'compact streaming'}
    : {exact_streaming:'精确流式', compact_streaming:'紧凑流式'};
  if (receipt && labels[receipt.selected_scheme]) {
    return `${receipt.resource_profile || (isEnglish() ? 'Auto VRAM' : '自动显存')}→${labels[receipt.selected_scheme]}`;
  }
  return isEnglish() ? 'Automatic VRAM optimization' : '显存自动优化';
}

function runtimeMemorySummary(job) {
  const memory = job?.inference_plan?.runtime_memory || {};
  const peak = Number(memory.peak_reserved_gib || memory.peak_allocated_gib);
  const ceiling = Number(memory.allocator_ceiling_gib);
  return Number.isFinite(peak) && peak > 0
    ? isEnglish()
      ? ` · reserved peak ${peak.toFixed(2)}GiB${Number.isFinite(ceiling) ? ` / hard limit ${ceiling.toFixed(2)}GiB` : ''}`
      : ` · 保留峰值 ${peak.toFixed(2)}GiB${Number.isFinite(ceiling) ? ` / 硬上限 ${ceiling.toFixed(2)}GiB` : ''}`
    : '';
}

function advancedSummary(req, job=null) {
  const english = isEnglish();
  const memory = memoryExecutionSummary(req, job);
  const selfLiftTail = req.selflift_enabled
    ? req.selflift_temporal_window_enabled
      ? english
        ? ` · windowed ${Number(req.selflift_temporal_window_seconds).toFixed(1)}s / overlap ${Number(req.selflift_temporal_overlap_seconds).toFixed(1)}s · Sigma ${Number(req.selflift_sigma_scale ?? 1).toFixed(2)}`
        : ` · 二采分窗 ${Number(req.selflift_temporal_window_seconds).toFixed(1)}秒 / 重叠 ${Number(req.selflift_temporal_overlap_seconds).toFixed(1)}秒 · Sigma ${Number(req.selflift_sigma_scale ?? 1).toFixed(2)}`
      : (english ? ' · full-timeline second pass' : ' · 整段二采')
    : '';
  if (req.sampling_steps != null && req.acceleration != null) {
    const totalSteps = Number(req.sampling_steps);
    const acceleration = Number(req.acceleration);
    const transitionStep = Math.max(0, Math.min(
      totalSteps,
      Number(req.acceleration_transition_step ?? totalSteps),
    ));
    const secondAcceleration = Number(req.second_pass_acceleration ?? acceleration);
    const accelerationLabel = value => value === 0 ? 'Dense' : `${english ? 'acceleration' : '加速'} ${value}`;
    if (transitionStep < totalSteps) {
      return english
        ? `${totalSteps} total steps · first ${transitionStep} steps ${accelerationLabel(acceleration)} / final ${totalSteps - transitionStep} steps ${accelerationLabel(secondAcceleration)} · ${memory}${selfLiftTail}`
        : `${totalSteps}总步 · 一采${transitionStep}步 ${accelerationLabel(acceleration)} / 二采${totalSteps - transitionStep}步 ${accelerationLabel(secondAcceleration)} · ${memory}${selfLiftTail}`;
    }
    return english
      ? `${totalSteps} total steps · ${accelerationLabel(acceleration)} · ${memory}${selfLiftTail}`
      : `${totalSteps}总步 · ${accelerationLabel(acceleration)} · ${memory}${selfLiftTail}`;
  }
  if (!req.advanced) return `${req.quality || (english ? 'Default compute' : '默认计算')} · ${memory}`;
  const compute = req.model_variant === 'base'
    ? english ? `${req.actual_steps} actual / ${req.forecast_steps} forecast` : `${req.actual_steps}实际/${req.forecast_steps}预测`
    : english ? `${req.lora_steps} steps` : `${req.lora_steps}步`;
  const attention = Number(req.attention_keep_ratio) >= 1
    ? english ? 'Full attention' : '完整注意力'
    : english
      ? `${Math.round(Number(req.attention_keep_ratio) * 100)}% attention · ${{full:'fixed throughout',guarded:'dynamic protection',middle_only:'middle only'}[req.sparse_scope] || req.sparse_scope}`
      : `${Math.round(Number(req.attention_keep_ratio) * 100)}%注意力 · ${{full:'全程固定',guarded:'动态保护',middle_only:'仅中段'}[req.sparse_scope] || req.sparse_scope}`;
  return `${compute} · ${attention} · ${memory}`;
}

function jobCard(job, {draggable=false, selectable=false}={}) {
  const english = isEnglish();
  const req = job.request;
  const second = job.second_sampling || null;
  const repair = job.video_repair || null;
  const cancelling = job.progress?.stage === 'cancelling';
  const canCancel = !cancelling && ['queued','starting_backend','running','awaiting_preview'].includes(job.status);
  const canDelete = !['starting_backend','running','awaiting_preview'].includes(job.status);
  const previewLabel = req.execution_mode === 'checkpoint'
    ? '查看断点预览'
    : job.infinite_continuation
      ? '查看尾段中途预览'
      : '查看中途预览';
  const actions = [
    job.preview?.ready ? `<button data-view-preview="${job.id}">${previewLabel}</button>` : '',
    job.checkpoint?.resume_available ? `<button data-resume="${job.id}">继续正式生成</button>` : '',
    job.status === 'awaiting_preview' ? `<button data-preview-continue="${job.id}">继续正式生成</button><button class="danger-action" data-preview-discard="${job.id}">放弃本次抽卡</button>` : '',
    job.video_repair_available ? `<button data-video-repair="${job.id}">人脸修复</button>` : '',
    job.status === 'succeeded' ? `<button data-view="${job.id}">预览与下载</button>` : '',
    canCancel ? `<button data-cancel="${job.id}">取消</button>` : '',
    canDelete ? `<button class="danger-action" data-delete="${job.id}">删除</button>` : '',
  ].join('');
  const elapsedDetail = repair
    ? `${english ? 'H3 Base face repair' : 'H3 Base 人脸修复'}${runtimeMemorySummary(job)}`
    : second
      ? `${english ? (second.method === 'temporal' ? 'Temporal-model second sampling' : 'H3 generative second sampling') : (second.method === 'temporal' ? '时序模型二采' : 'H3 生成式二采')}${runtimeMemorySummary(job)}`
      : job.upscale_elapsed_seconds != null
        ? `${english ? 'Historical H3' : '历史 H3'} ${formatElapsed(job.generation_elapsed_seconds)} · FlashVSR ${formatElapsed(job.upscale_elapsed_seconds)}`
        : `${english ? 'Excludes service startup and model preload' : '不含服务启动与模型预加载'}${runtimeMemorySummary(job)}`;
  const elapsed = job.status === 'succeeded'
    ? `<div class="job-elapsed"><span>${english ? 'Actual total time' : '实际总耗时'}</span><strong>${formatElapsed(job.elapsed_seconds)}</strong><small>${elapsedDetail}</small></div>`
    : '';
  const delivery = repair
    ? ` · ${english ? 'repaired from' : '由'} ${escapeHtml(repair.source_job_id || (english ? 'source job' : '源任务'))}${english ? '' : ' 修复'}`
    : second ? ` · ${english ? 'second-sampled from' : '由'} ${escapeHtml(second.source_job_id || (english ? 'source job' : '源任务'))}${english ? '' : ' 二次采样'}` : '';
  const execution = repair
    ? english
      ? `Face repair · ${repair.canvas_size || 768}P square canvas · ${repair.capacity || repair.max_faces || 4} cells · at least ${Number(repair.minimum_magnification || 1.5).toFixed(1)}× · four-step Turbo · acceleration ${Number(repair.acceleration ?? 50)}`
      : `人脸修复 · ${repair.canvas_size || 768}P 方形画布 · ${repair.capacity || repair.max_faces || 4} Cell · 至少 ${Number(repair.minimum_magnification || 1.5).toFixed(1)}× · 四步 Turbo · 加速 ${Number(repair.acceleration ?? 50)}`
    : second
    ? (second.method === 'temporal'
      ? `${english ? 'One-step temporal video diffusion' : '一步时序视频扩散'} · ${memoryExecutionSummary(req, job)}`
      : english
        ? `${second.steps} second-pass steps · acceleration ${Number(second.acceleration)} · ${memoryExecutionSummary(req, job)}`
        : `${second.steps}二采实际步 · 加速 ${Number(second.acceleration)} · ${memoryExecutionSummary(req, job)}`)
    : advancedSummary(req, job);
  const selected = selectable && selectedHistoryJobIds.has(job.id);
  const selector = selectable
    ? `<label class="job-select" title="选择这条历史任务"><input type="checkbox" data-history-select="${job.id}" aria-label="选择这条历史任务" ${selected ? 'checked' : ''}></label>`
    : '';
  return `<article class="job manager-job ${draggable ? 'draggable' : ''} ${selectable ? 'selectable' : ''} ${selected ? 'selected' : ''}" data-job-id="${job.id}" ${draggable ? 'draggable="true"' : ''}>
    ${selector}
    <div class="job-top"><div class="job-main">${draggable ? `<span class="drag-handle" title="${english ? 'Drag to reorder' : '拖动排序'}">⠿</span>` : ''}<div><div class="job-title">${escapeHtml(req.prompt)}</div><div class="job-meta">${req.width}×${req.height}${delivery} · ${req.actual_duration_seconds.toFixed(2)}${english ? 's' : '秒'} · ${escapeHtml(execution)} · Seed ${req.seed}</div></div></div><span class="status ${job.status}">${statusName(job)}</span></div>
    ${progressMarkup(job)}${elapsed}${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ''}${actions ? `<div class="job-actions">${actions}</div>` : ''}
  </article>`;
}

function promptPreview(prompt) {
  const body = String(prompt || '')
    .replace(/^(integrated_multimodal_description|overall_soundscape|non_diegetic_music):\s*/gmi, '')
    .replace(/\[Shot \d+\](?: At [^,]+,)?/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return body.length > 420 ? `${body.slice(0, 420)}…` : body;
}

function conversationItem(job) {
  const req = job.request;
  const canCancel = ['queued','starting_backend','running','awaiting_preview'].includes(job.status);
  const canDelete = !['starting_backend','running','awaiting_preview'].includes(job.status);
  const completed = job.status === 'succeeded';
  const checkpointed = job.status === 'checkpointed';
  const previewLabel = req.execution_mode === 'checkpoint' ? '查看断点预览' : '查看中途预览';
  const output = completed
    ? `<div class="conversation-result"><button class="conversation-video-placeholder" data-view="${job.id}" aria-label="打开成片预览"><span>▶</span><small>点击预览成片</small></button><div><strong>${job.video_repair ? '人脸修复完成' : job.second_sampling ? (job.second_sampling.method === 'temporal' ? '时序模型二采完成' : 'H3 生成式二采完成') : '视频生成完成'}</strong><span>${req.width}×${req.height} · ${req.actual_duration_seconds.toFixed(2)}秒 · ${formatElapsed(job.elapsed_seconds)}</span><div class="conversation-actions"><button data-view="${job.id}">打开预览与下载</button></div></div></div>`
    : checkpointed
      ? `<div class="conversation-response-state"><span class="conversation-spinner stopped"></span><div><strong>已在第 ${job.checkpoint?.completed_steps || '?'} / ${job.checkpoint?.total_steps || '?'} 步停止</strong><small>正式状态已落盘，当前任务不占用 GPU；恢复时重新进入队列。</small></div></div>`
    : `<div class="conversation-response-state"><span class="conversation-spinner ${['failed','cancelled'].includes(job.status) ? 'stopped' : ''}"></span><div><strong>${escapeHtml(statusName(job))}</strong><small>${escapeHtml(job.progress?.detail || '')}</small></div></div>`;
  return `<article class="conversation-turn" data-conversation-job="${job.id}">
    <div class="conversation-user"><div class="bubble-label">你提交的视频任务</div><p>${escapeHtml(promptPreview(req.prompt))}</p><small>${req.width}×${req.height} · ${req.actual_duration_seconds.toFixed(2)}秒 · Seed ${req.seed}</small></div>
    <div class="conversation-assistant"><div class="bubble-label">H3 · ${req.model_variant === 'lora' ? 'LoRA 极速' : '原始权重'}</div>${output}${job.preview?.ready || job.checkpoint?.resume_available ? `<div class="conversation-actions">${job.preview?.ready ? `<button data-view-preview="${job.id}">${previewLabel}</button>` : ''}${job.checkpoint?.resume_available ? `<button data-resume="${job.id}">继续正式生成</button>` : ''}${job.status === 'awaiting_preview' ? `<button data-preview-continue="${job.id}">继续正式生成</button><button data-preview-discard="${job.id}">放弃抽卡</button>` : ''}</div>` : ''}${progressMarkup(job)}${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ''}<div class="conversation-record-actions">${canCancel ? `<button data-cancel="${job.id}">取消任务</button>` : ''}${canDelete ? `<button class="danger-action" data-delete="${job.id}">删除这条创作记录</button>` : ''}</div></div>
  </article>`;
}

function renderConversation() {
  const feed = $('#conversationFeed');
  if (!feed) return;
  const ordered = [...jobs].sort((a, b) => Number(a.created_at) - Number(b.created_at));
  feed.innerHTML = ordered.length
    ? ordered.map(conversationItem).join('')
    : '<div class="conversation-welcome"><b>从底部开始创建第一条视频</b><span>提交后，排队、生成进度、预计完成时间和成片都会显示在这里。</span></div>';
}

function empty(message) { return `<div class="manager-empty">${message}</div>`; }

function renderJobs() {
  const running = jobs.filter(job => ['starting_backend','running','awaiting_preview'].includes(job.status));
  const queued = jobs.filter(job => job.status === 'queued').sort((a,b) => (a.queue_position || 999) - (b.queue_position || 999));
  const history = jobs.filter(job => !['queued','starting_backend','running','awaiting_preview'].includes(job.status));
  const historyIds = new Set(history.map(job => job.id));
  for (const id of selectedHistoryJobIds) {
    if (!historyIds.has(id)) selectedHistoryJobIds.delete(id);
  }
  $('#runningCount').textContent = running.length;
  $('#queuedCount').textContent = queued.length;
  $('#completedCount').textContent = history.filter(job => job.status === 'succeeded').length;
  $('#navTaskCount').textContent = running.length + queued.length;
  $('#runningJobs').innerHTML = running.length ? running.map(job => jobCard(job)).join('') : empty('当前没有正在执行的任务');
  $('#queuedJobs').innerHTML = queued.length ? queued.map(job => jobCard(job, {draggable:true})).join('') : empty('等待队列为空');
  $('#historyJobs').innerHTML = history.length ? history.map(job => jobCard(job, {selectable:true})).join('') : empty('还没有历史任务');
  updateHistorySelectionToolbar(history.length);
  renderConversation();
  bindJobActions(); bindDragAndDrop();
}

function bindJobActions() {
  $$('[data-history-select]').forEach(input => input.addEventListener('change', () => toggleHistorySelection(input.dataset.historySelect, input.checked)));
  $$('[data-cancel]').forEach(button => button.addEventListener('click', () => cancelJob(button.dataset.cancel)));
  $$('[data-delete]').forEach(button => button.addEventListener('click', () => deleteJob(button.dataset.delete)));
  $$('[data-view]').forEach(button => button.addEventListener('click', () => showVideo(button.dataset.view)));
  $$('[data-view-preview]').forEach(button => button.addEventListener('click', () => showPreview(button.dataset.viewPreview)));
  $$('[data-video-repair]').forEach(button => button.addEventListener('click', () => openVideoRepair(button.dataset.videoRepair)));
  $$('[data-resume]').forEach(button => button.addEventListener('click', () => resumeJob(button.dataset.resume)));
  $$('[data-preview-continue]').forEach(button => button.addEventListener('click', () => decidePreview(button.dataset.previewContinue, 'continue')));
  $$('[data-preview-discard]').forEach(button => button.addEventListener('click', () => decidePreview(button.dataset.previewDiscard, 'discard')));
}

function historyJobs() {
  return jobs.filter(job => !['queued','starting_backend','running','awaiting_preview'].includes(job.status));
}

function updateHistorySelectionToolbar(historyCount=historyJobs().length) {
  const selectedCount = selectedHistoryJobIds.size;
  $('#historySelectionCount').textContent = window.H3I18n?.locale === 'en'
    ? `${selectedCount} selected`
    : `已选择 ${selectedCount} 项`;
  $('#selectAllHistory').disabled = historyCount === 0 || selectedCount === historyCount;
  $('#clearHistorySelection').disabled = selectedCount === 0;
  $('#deleteSelectedHistory').disabled = selectedCount === 0;
}

function toggleHistorySelection(id, selected) {
  if (selected) selectedHistoryJobIds.add(id);
  else selectedHistoryJobIds.delete(id);
  const card = $(`#historyJobs [data-job-id="${id}"]`);
  if (card) card.classList.toggle('selected', selected);
  updateHistorySelectionToolbar();
}

function selectAllHistory() {
  for (const job of historyJobs()) selectedHistoryJobIds.add(job.id);
  renderJobs();
}

function clearHistorySelection() {
  selectedHistoryJobIds.clear();
  renderJobs();
}

function bindDragAndDrop() {
  $$('#queuedJobs [draggable="true"]').forEach(card => {
    card.addEventListener('dragstart', event => { draggedJobId = card.dataset.jobId; card.classList.add('dragging'); event.dataTransfer.effectAllowed = 'move'; });
    card.addEventListener('dragend', () => { card.classList.remove('dragging'); draggedJobId = null; });
    card.addEventListener('dragover', event => { event.preventDefault(); const dragged = $(`#queuedJobs [data-job-id="${draggedJobId}"]`); if (!dragged || dragged === card) return; const rect = card.getBoundingClientRect(); card.parentElement.insertBefore(dragged, event.clientY < rect.top + rect.height / 2 ? card : card.nextSibling); });
    card.addEventListener('drop', async event => { event.preventDefault(); await saveQueueOrder(); });
  });
}

async function saveQueueOrder() {
  const jobIds = $$('#queuedJobs [data-job-id]').map(card => card.dataset.jobId);
  try { await api('/api/v1/jobs/order', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_ids:jobIds})}); await refreshJobs(); }
  catch (error) { alert(tr(`调整顺序失败：${error.message}`)); await refreshJobs(); }
}

async function refreshJobs() {
  try { jobs = (await (await api('/api/v1/jobs?limit=100')).json()).jobs; renderJobs(); }
  catch (error) { $('#healthText').textContent = error.message; }
}

async function checkHealth() {
  try {
    const response = await api('/healthz', {timeoutMs:5000}); const health = await response.json();
    $('.server-state').className = 'server-state online';
    const warm = health.warm_state?.status || 'unknown';
    renderEngineLoadProgress(health.warm_state, health.engine_control?.switching);
    const warmName = isEnglish()
      ? {cold:'Not loaded',loading:'Loading',ready:'Warm',failed:'Load failed',unsupported:'Available'}[warm] || warm
      : {cold:'未加载',loading:'加载中',ready:'已热身',failed:'加载失败',unsupported:'可用'}[warm] || warm;
    $('#warmStateText').textContent = warmName;
    const engineLabel = health.active_engine === 'reference' || health.active_engine === 'reference_lora'
      ? 'Ref2VA' : health.active_engine === 'first_last' || health.active_engine === 'original' || health.active_engine === 'lora'
        ? 'FL2VA' : isEnglish() ? 'Choose a mode' : '待选择模式';
    $('#healthText').textContent = isEnglish()
      ? health.engine_control?.switching ? 'Online · switching engine' : warm === 'loading' ? 'Online · preloading model' : warm === 'failed' ? 'Online · model load failed' : `Online · ${engineLabel} · ${warmName}`
      : health.engine_control?.switching ? '在线 · 正在切换引擎' : warm === 'loading' ? '在线 · 正在预加载模型' : warm === 'failed' ? '在线 · 模型加载失败' : `在线 · ${engineLabel} · ${warmName}`;
  } catch (_) {
    $('.server-state').className = 'server-state offline';
    $('#healthText').textContent = isEnglish() ? 'Service unavailable' : '服务不可用';
    $('#warmStateText').textContent = isEnglish() ? 'Offline' : '离线';
  }
}

async function pollUiState() {
  if (document.hidden || uiPollPromise) return uiPollPromise;
  uiPollPromise = (async () => {
    await Promise.allSettled([checkHealth(), refreshJobs()]);
    const lobbyLoading = $('#engineLobby').classList.contains('loading');
    const needsReconcile = !options || !currentEngine
      || Boolean(options?.engine_control?.switching) || lobbyLoading;
    if (needsReconcile) await reconcileEngineState().catch(() => false);
  })();
  try { return await uiPollPromise; }
  finally { uiPollPromise = null; }
}

function setResourceBar(id, percent) {
  const bar = $(id); if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
}

async function refreshResources() {
  if (!currentEngine || activePage !== 'tasks') return;
  try {
    const data = await (await api('/api/v1/resources')).json();
    const hostMemory = data.memory;
    const hostOccupied = hostMemory.occupied_gib ?? hostMemory.used_gib;
    const hostOccupiedPercent = hostMemory.occupied_percent ?? hostMemory.percent;
    const hostReclaimable = hostMemory.reclaimable_gib ?? 0;
    $('#hostMemoryUsage').textContent = `${hostOccupied.toFixed(1)} / ${hostMemory.total_gib.toFixed(1)} GiB`;
    $('#hostMemoryDetail').textContent = window.H3I18n?.locale === 'en'
      ? `${hostMemory.available_gib.toFixed(1)} GiB available · ${hostReclaimable.toFixed(1)} GiB reclaimable cache`
      : `${hostMemory.available_gib.toFixed(1)} GiB 可用 · ${hostReclaimable.toFixed(1)} GiB 可回收缓存`;
    setResourceBar('#hostMemoryBar', hostOccupiedPercent);
    const serviceMemory = data.service_memory;
    if (serviceMemory) {
      const serviceResident = serviceMemory.resident_gib ?? data.process?.rss_gib ?? serviceMemory.used_gib;
      const serviceProcessCount = Number(serviceMemory.resident_process_count) || 1;
      const serviceHostPercent = hostMemory.total_gib > 0
        ? 100 * serviceResident / hostMemory.total_gib
        : 0;
      $('#memoryUsage').textContent = `${serviceResident.toFixed(1)} GiB`;
      $('#memoryDetail').textContent = window.H3I18n?.locale === 'en'
        ? `${serviceHostPercent.toFixed(0)}% of host · ${serviceProcessCount} service/worker processes · model mappings included`
        : `占整机 ${serviceHostPercent.toFixed(0)}% · ${serviceProcessCount} 个服务/推理进程 · 含模型映射`;
      setResourceBar('#memoryBar', serviceHostPercent);
    } else {
      $('#memoryUsage').textContent = `${data.process.rss_gib.toFixed(1)} GiB`;
      $('#memoryDetail').textContent = window.H3I18n?.locale === 'en'
        ? 'H3 process · no active memory limit'
        : 'H3服务进程 · 尚未分配内存上限';
      setResourceBar('#memoryBar', 0);
    }
    if (data.gpu) {
      $('#gpuUsage').textContent = `${data.gpu.utilization_percent.toFixed(0)}%`;
      $('#gpuDetail').textContent = `${data.gpu.name} · ${data.gpu.temperature_c.toFixed(0)}°C · ${data.gpu.power_w.toFixed(0)}W`;
      setResourceBar('#gpuBar', data.gpu.utilization_percent);
      $('#vramUsage').textContent = `${data.gpu.memory_used_gib.toFixed(1)} / ${data.gpu.memory_total_gib.toFixed(1)} GiB`;
      $('#vramDetail').textContent = window.H3I18n?.locale === 'en'
        ? `${data.gpu.memory_percent.toFixed(0)}% used`
        : `${data.gpu.memory_percent.toFixed(0)}% 已使用`;
      setResourceBar('#vramBar', data.gpu.memory_percent);
    } else {
      $('#gpuUsage').textContent = isEnglish() ? 'Unavailable' : '不可用';
      $('#gpuDetail').textContent = isEnglish() ? 'No NVIDIA monitoring interface detected' : '未检测到 NVIDIA 监控接口';
      $('#vramUsage').textContent = isEnglish() ? 'Unavailable' : '不可用';
      $('#vramDetail').textContent = isEnglish() ? 'nvidia-smi is not ready' : 'nvidia-smi 未就绪';
    }
  } catch (_) {}
}

async function cancelJob(id) {
  const button = document.querySelector(`[data-cancel="${id}"]`);
  if (button) { button.disabled = true; button.textContent = '正在取消…'; }
  try { await api(`/api/v1/jobs/${id}`, {method:'DELETE'}); await refreshJobs(); }
  catch (error) { if (button) button.disabled = false; alert(tr(error.message)); }
}
async function deleteJob(id) { if (!confirm(tr('删除该任务记录、上传帧和已生成视频？此操作不可恢复。'))) return; try { await api(`/api/v1/jobs/${id}/record`, {method:'DELETE'}); selectedHistoryJobIds.delete(id); await refreshJobs(); } catch (error) { alert(tr(error.message)); } }

async function deleteSelectedHistory() {
  const ids = [...selectedHistoryJobIds];
  if (!ids.length) return;
  const english = window.H3I18n?.locale === 'en';
  const warning = english
    ? `Delete ${ids.length} selected job records, uploads, generated videos, checkpoints, and latents? This cannot be undone.`
    : `删除已选择的 ${ids.length} 条任务记录、上传素材、成片、断点和 Latent？此操作不可恢复。`;
  if (!confirm(warning)) return;
  const button = $('#deleteSelectedHistory');
  button.disabled = true;
  button.textContent = tr('正在批量删除…');
  try {
    const response = await api('/api/v1/jobs/records', {
      method:'DELETE',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({job_ids:ids}),
    });
    const result = await response.json();
    for (const id of result.deleted_ids || []) selectedHistoryJobIds.delete(id);
    await refreshJobs();
    if (result.errors?.length) {
      const detail = result.errors.map(item => `${String(item.id).slice(0, 8)}: ${item.error}`).join('\n');
      alert(english
        ? `${result.deleted_count} deleted; ${result.errors.length} failed:\n${detail}`
        : `已删除 ${result.deleted_count} 条，${result.errors.length} 条失败：\n${detail}`);
    }
  } catch (error) {
    alert(tr(error.message));
  } finally {
    button.textContent = tr('删除已选');
    updateHistorySelectionToolbar();
  }
}
async function showVideo(id) { try { const blob = await (await api(`/api/v1/jobs/${id}/video`)).blob(); if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl = URL.createObjectURL(blob); $('#resultVideo').src = videoObjectUrl; $('#downloadVideo').href = videoObjectUrl; $('#downloadVideo').download = `h3-${id}.mp4`; $('#videoDialog').showModal(); } catch (error) { alert(tr(error.message)); } }
async function showPreview(id) { try { const blob = await (await api(`/api/v1/jobs/${id}/preview`)).blob(); if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl = URL.createObjectURL(blob); $('#resultVideo').src = videoObjectUrl; $('#downloadVideo').href = videoObjectUrl; $('#downloadVideo').download = `h3-${id}-preview.mp4`; $('#videoDialog').showModal(); } catch (error) { alert(tr(error.message)); } }
async function decidePreview(id, decision) { try { await api(`/api/v1/jobs/${id}/preview/${decision}`, {method:'POST'}); await refreshJobs(); } catch (error) { alert(tr(error.message)); } }
async function resumeJob(id) { try { await api(`/api/v1/jobs/${id}/resume`, {method:'POST'}); await refreshJobs(); } catch (error) { alert(tr(error.message)); } }

function updateVideoRepairSummary() {
  const acceleration = Number($('#videoRepairAcceleration').value);
  $('#videoRepairAccelerationValue').textContent = String(acceleration);
  $('#videoRepairGeometrySummary').textContent = window.H3I18n?.locale === 'en'
    ? `Uses the global ${globalFaceRepairPolicy.canvas_size}P square canvas with ${globalFaceRepairPolicy.capacity} cells and fixed four-step Turbo repair.`
    : `使用全局 ${globalFaceRepairPolicy.canvas_size}P 方形画布与 ${globalFaceRepairPolicy.capacity} 个 Cell，固定四步 Turbo 修复。`;
}

function openVideoRepair(id) {
  const job = jobs.find(item => item.id === id);
  if (!job || !job.video_repair_available) return;
  $('#videoRepairSourceId').value = id;
  $('#videoRepairSourceSummary').textContent = window.H3I18n?.locale === 'en'
    ? `Source video ${job.request.width}×${job.request.height} · ${job.request.actual_duration_seconds.toFixed(2)} sec; repair preserves its resolution and audio.`
    : `源视频 ${job.request.width}×${job.request.height} · ${job.request.actual_duration_seconds.toFixed(2)}秒；修复后保持原分辨率和原音频。`;
  $('#videoRepairMessage').hidden = true;
  updateVideoRepairSummary();
  $('#videoRepairDialog').showModal();
}

async function submitVideoRepair(event) {
  event.preventDefault();
  const id = $('#videoRepairSourceId').value;
  const message = $('#videoRepairMessage');
  const button = $('button[type="submit"]', event.target);
  button.disabled = true;
  message.hidden = false;
  message.textContent = tr('正在创建人脸修复任务…');
  try {
    const response = await api(`/api/v1/jobs/${id}/video-repair`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        acceleration:Number($('#videoRepairAcceleration').value),
      }),
    });
    jobs.push(await response.json());
    $('#videoRepairDialog').close();
    renderJobs();
    switchPage('tasks');
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function clearLatentCache() {
  if (!confirm(tr('清理所有Latent与断点缓存？成片、任务记录和基于成片的视频修复不受影响；长视频最终采样与断点续跑将失去所需缓存。'))) return;
  const button = $('#clearLatentCache');
  button.disabled = true;
  try {
    const response = await api('/api/v1/cache/latents', {method:'DELETE'});
    const result = await response.json();
    const mib = Number(result.removed_bytes || 0) / 1024 / 1024;
    alert(tr(`已清理 ${result.removed_files || 0} 个Latent文件，共 ${mib.toFixed(1)} MiB。`));
    await refreshJobs();
  } catch (error) {
    alert(tr(error.message));
  } finally {
    button.disabled = false;
  }
}

async function submit(event) {
  event.preventDefault();
  const button = $('.submit-button', event.target), message = $('#formMessage');
  button.dataset.submitting = 'true';
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  button.textContent = '…';
  message.textContent = '正在检查参数并提交任务…';
  message.style.background = 'rgba(138,120,255,.10)';
  message.style.color = '#d4ceff';
  message.hidden = false;
  try {
    if (!bootReady || !options) throw new Error('控制台仍在初始化，请稍候后重试');
    if (!currentEngine) throw new Error('请先选择生成模式');
    const maximumDuration = currentMaxDuration();
    const requested = Number(selected('duration_seconds')) || 0;
    if (requested < 1 || requested > maximumDuration) throw new Error(`当前分辨率的视频时长必须在 1–${maximumDuration} 秒之间`);
    if (!$('#freeformPrompt').value.trim()) throw new Error('请输入完整提示词');
    const form = new FormData(event.target); form.delete('engine');
    form.set('service_family', currentEngine);
    form.set('model_variant', currentVariant());
    form.set('execution_mode', 'complete');
    ['checkpoint_step','checkpoint_retain','checkpoint_preview','checkpoint_preview_steps','checkpoint_preview_resolution'].forEach(name => form.delete(name));
    const totalSteps = solverStepCount();
    const firstSteps = Math.max(1, Math.min(totalSteps, Number($('#firstPassSteps').value) || totalSteps));
    const secondPassAcceleration = Math.max(0, Math.min(100, Number($('[name="second_pass_acceleration"]').value) || 0));
    form.set('second_pass_acceleration', String(secondPassAcceleration));
    form.set('acceleration_transition_step', String(firstSteps));
    if ($('#previewEnabled').checked) {
      form.set('preview_mode', 'pause');
      form.set('preview_step_index', String(firstSteps - 1));
      form.delete('preview_branch_steps');
      form.set('preview_fast_finish', 'true');
      form.set('checkpoint_preview_resolution', 'source');
    } else {
      ['preview_mode','preview_step_index','preview_branch_steps','preview_fast_finish'].forEach(name => form.delete(name));
    }
    if (currentEngine === 'reference') {
      const files = Array.from($('#referenceImages').files || []);
      const videos = Array.from($('#referenceVideos').files || []);
      const audios = Array.from($('#referenceAudios').files || []);
      if (!files.length && !videos.length && !audios.length) throw new Error('请至少选择一项参考图片、视频或音频');
      if (files.length > 9) throw new Error('参考图片最多9张');
      if (videos.length > 3) throw new Error('参考视频最多3段');
      if (audios.length > 3) throw new Error('参考音频最多3段');
      const bindingError = referenceBindingError();
      if (bindingError) throw new Error(bindingError);
      files.forEach((file, index) => form.set(`reference_image_${index + 1}`, file));
      videos.forEach((file, index) => form.set(`reference_video_${index + 1}`, file));
      audios.forEach((file, index) => form.set(`reference_audio_${index + 1}`, file));
    }
    form.set('mode', 'preset');
    form.set('advanced', 'false');
    ['width','height','size_mode'].forEach(name => form.delete(name));
    ['frames','advanced_seed','actual_steps','lora_steps','attention_keep_ratio','sparse_scope'].forEach(name => form.delete(name));
    const initialResolution = Number.parseInt($('#firstPassResolution').value, 10);
    const targetResolution = Number.parseInt(selected('resolution'), 10);
    if (initialResolution < targetResolution) {
      if (firstSteps >= totalSteps) {
        throw new Error('渐进分辨率至少需要保留一步用于最终分辨率生成');
      }
      form.set('selflift_enabled', 'true');
      form.set('selflift_initial_resolution', $('#firstPassResolution').value);
      form.set('selflift_transition_step', String(firstSteps));
      const windowed = Boolean($('#singleSecondSamplingWindowEnabled')?.checked);
      form.set('selflift_temporal_window_enabled', windowed ? 'true' : 'false');
      if (windowed) {
        form.set(
          'selflift_sigma_scale',
          String(Math.max(0.25, Math.min(1,
            Number($('#singleSecondSamplingSigmaScale')?.value) || 1,
          ))),
        );
      } else {
        form.delete('selflift_sigma_scale');
      }
    } else {
      ['selflift_enabled','selflift_initial_resolution','selflift_transition_step','selflift_sigma_scale'].forEach(name => form.delete(name));
      form.set('selflift_temporal_window_enabled', 'false');
    }
    for (const role of ['first_frame','last_frame']) if (!$(`[name="${role}"]`).files[0]) form.delete(role);
    const job = await (await api('/api/v1/generations', {method:'POST', body:form})).json(); jobs.push(job); renderJobs(); message.textContent = `任务 ${job.id.slice(0,8)} 已加入队列`; message.style.background = 'rgba(86,214,160,.1)'; message.style.color = '#8ce9bd'; message.hidden = false; $('#conversationFeed').lastElementChild?.scrollIntoView({behavior:'smooth', block:'center'});
  } catch (error) {
    message.textContent = error.message;
    message.style.background = '';
    message.style.color = '';
    message.hidden = false;
    message.scrollIntoView({behavior:'smooth', block:'nearest'});
  } finally {
    delete button.dataset.submitting;
    button.removeAttribute('aria-busy');
    button.textContent = '↑';
    updateSubmitAvailability();
  }
}

async function boot() {
  bindDropzone('#firstDrop'); bindDropzone('#lastDrop'); $('#generationForm').addEventListener('submit', submit);
  $('#freeformPrompt').addEventListener('input', event => { updateReferenceMentionMenu($('#freeformPromptEditor'), event.target); updateContract(); });
  $('#freeformPrompt').addEventListener('keydown', event => { if (event.key === 'Escape') $('.reference-mention-menu', $('#freeformPromptEditor')).hidden = true; });
  $('#freeformPrompt').addEventListener('blur', () => setTimeout(() => { $('.reference-mention-menu', $('#freeformPromptEditor')).hidden = true; }, 120));
  $('[name="model_variant"]').addEventListener('change', () => { applyEngineIdentity(); updateContract(); });
  $('#loraAccelerationEnabled').addEventListener('change', event => {
    const variant = $('[name="model_variant"]');
    variant.value = event.target.checked ? 'lora' : 'base';
    variant.dispatchEvent(new Event('change', {bubbles:true}));
  });
  $('#referenceFiles').addEventListener('change', event => distributeReferenceFiles(event.target.files));
  bindFileDrop($('#referenceFileDrop'), $('#referenceFiles'), {maxFiles:15, accept:file => file.type.startsWith('image/') || file.type.startsWith('video/') || file.type.startsWith('audio/') || /\.(mp4|mov|mkv|webm|avi|wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name)});
  $('#referenceImages').addEventListener('change', event => {
    renderReferencePreviews('image');
    updateContract();
  });
  $('#referenceVideos').addEventListener('change', event => {
    renderReferencePreviews('video');
    updateContract();
  });
  $('#referenceAudios').addEventListener('change', event => {
    renderReferencePreviews('audio');
    updateContract();
  });
  $('#exitEngine').addEventListener('click', exitEngine);
  $('#chooseWorkspace').addEventListener('click', openWorkspaceDialog);
  $('#openWorkspacePath').addEventListener('click', () => browseWorkspace($('#workspacePathInput').value.trim()).catch(showWorkspaceError));
  $('#workspacePathInput').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); browseWorkspace(event.target.value.trim()).catch(showWorkspaceError); } });
  $('#workspaceParent').addEventListener('click', () => { if (workspaceBrowseParent) browseWorkspace(workspaceBrowseParent).catch(showWorkspaceError); });
  $('#workspaceDefault').addEventListener('click', () => browseWorkspace(options.workspace.default_path).catch(showWorkspaceError));
  $('#selectWorkspace').addEventListener('click', activateWorkspace);
  $('#videoRepairForm').addEventListener('submit', submitVideoRepair);
  $('#closeVideoRepair').addEventListener('click', () => $('#videoRepairDialog').close());
  $('#videoRepairAcceleration').addEventListener('input', updateVideoRepairSummary);
  $('#clearLatentCache').addEventListener('click', clearLatentCache);
  $$('.workspace-tabs button').forEach(button => button.addEventListener('click', () => switchPage(button.dataset.page)));
  $('#creationResolutionSlider').addEventListener('input', event => {
    setCreationResolution(event.target.value);
  });
  $$('#creationResolutionControl [data-resolution]').forEach(button => {
    button.addEventListener('click', () => setCreationResolution(button.dataset.resolution, true));
  });
  $('#generationForm [name="aspect_ratio"]').addEventListener('input', () => {
    renderCreationResolution();
    syncDurationControl();
  });
  $('[name="duration_seconds"]').addEventListener('input', () => {
    $('#freeformDurationValue').textContent = isEnglish()
      ? `${Number(selected('duration_seconds')).toFixed(1)}s`
      : `${Number(selected('duration_seconds')).toFixed(1)} 秒`;
    updateContract();
  });
  $('[name="sampling_steps"]').addEventListener('input', updateJointAccelerationControls);
  $('#firstPassSteps').addEventListener('input', updateSelfLiftControls);
  $('[name="acceleration"]').addEventListener('input', updateJointAccelerationControls);
  $('[name="second_pass_acceleration"]').addEventListener('input', updateJointAccelerationControls);
  $('#previewEnabled').addEventListener('change', updateSelfLiftControls);
  $('#singleSecondSamplingWindowEnabled').addEventListener('change', event => {
    event.target.dataset.userTouched = 'true';
    updateSelfLiftControls();
  });
  $('#singleSecondSamplingSigmaScale').addEventListener('input', updateSelfLiftControls);
  $('#firstPassResolutionSlider').addEventListener('input', event => {
    setFirstPassResolution(event.target.value);
  });
  $('#refreshJobs').addEventListener('click', refreshJobs);
  $('#selectAllHistory').addEventListener('click', selectAllHistory);
  $('#clearHistorySelection').addEventListener('click', clearHistorySelection);
  $('#deleteSelectedHistory').addEventListener('click', deleteSelectedHistory);
  $('#settingsButton').addEventListener('click', async () => {
    $('#apiKeyInput').value = localStorage.getItem('h3serve_api_key') || '';
    const [referencePolicy, loraPolicy, faceRepairPolicy, previewPolicy, secondSamplingWindowPolicy] = await Promise.all([
      serverReferenceMediaSettings().catch(() => null),
      serverLoraSettings().catch(() => null),
      serverFaceRepairSettings().catch(() => null),
      serverPreviewSettings().catch(() => null),
      serverSecondSamplingWindowSettings().catch(() => null),
    ]);
    if (referencePolicy) renderReferenceMediaPolicy(referencePolicy);
    if (loraPolicy) renderGlobalLoraPolicy(loraPolicy);
    if (faceRepairPolicy) renderFaceRepairPolicy(faceRepairPolicy);
    if (previewPolicy) renderPreviewPolicy(previewPolicy);
    if (secondSamplingWindowPolicy) renderSecondSamplingWindowPolicy(secondSamplingWindowPolicy);
    $('#settingsDialog').showModal();
  });
  $('#globalFaceRepairCanvas').addEventListener('input', updateFaceRepairSettingsOutputs);
  $('#globalPreviewSteps').addEventListener('input', event => renderPreviewPolicy({steps:event.target.value}));
  $('#globalSecondSamplingWindowEnabled').addEventListener('change', event => {
    renderSecondSamplingWindowPolicy({enabled:event.target.checked});
  });
  $('#globalSecondSamplingWindowSeconds').addEventListener('input', event => {
    renderSecondSamplingWindowPolicy({window_seconds:event.target.value});
  });
  $('#globalSecondSamplingOverlapSeconds').addEventListener('input', event => {
    renderSecondSamplingWindowPolicy({overlap_seconds:event.target.value});
  });
  $('#loadGlobalLora').addEventListener('click', loadSelectedGlobalLora);
  $('#saveSettings').addEventListener('click', async event => {
    event.preventDefault();
    const value = $('#apiKeyInput').value.trim();
    value ? localStorage.setItem('h3serve_api_key', value) : localStorage.removeItem('h3serve_api_key');
    try {
      const [referencePolicy, faceRepairPolicy, previewPolicy, secondSamplingWindowPolicy] = await Promise.all([
        configureServerReferenceMedia($('#globalReferenceImageResolution').value, $('#globalReferenceVideoResolution').value),
        configureServerFaceRepair($('#globalFaceRepairCanvas').value, $('#globalFaceRepairCapacity').value),
        configureServerPreview($('#globalPreviewSteps').value),
        configureServerSecondSamplingWindow(
          $('#globalSecondSamplingWindowEnabled').checked,
          $('#globalSecondSamplingWindowSeconds').value,
          $('#globalSecondSamplingOverlapSeconds').value,
        ),
      ]);
      renderReferenceMediaPolicy(referencePolicy);
      renderFaceRepairPolicy(faceRepairPolicy);
      renderPreviewPolicy(previewPolicy);
      renderSecondSamplingWindowPolicy(secondSamplingWindowPolicy);
      $('#settingsDialog').close();
    } catch (error) {
      $('#formMessage').textContent = error.message;
      $('#formMessage').hidden = false;
    }
    setTimeout(() => { reloadOptions().catch(() => {}); bootData(); }, 0);
  });
  try { await reloadOptions(); }
  catch (error) { $('#formMessage').textContent = error.message; $('#formMessage').hidden = false; }
  updateVideoRepairSummary();
  await bootData();
  syncDurationControl();
  bootReady = true;
  updateSubmitAvailability();
  setInterval(pollUiState, 2500);
  setInterval(() => { if (!document.hidden) refreshResources(); }, 1000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) { pollUiState(); refreshResources(); }
  });
}

async function bootData() { await Promise.allSettled([checkHealth(), refreshJobs(), refreshResources()]); }
window.addEventListener('h3serve:locale-changed', () => {
  if (!options) return;
  updateVideoRepairSummary();
  updateJointAccelerationControls();
  updateSelfLiftControls();
  renderReferenceMediaPolicy();
  renderReferencePreviews('image');
  renderReferencePreviews('video');
  renderReferencePreviews('audio');
  renderFaceRepairPolicy();
  renderSecondSamplingWindowPolicy();
  renderGlobalLoraPolicy();
  renderJobs();
  refreshResources().catch(() => {});
});
document.addEventListener('DOMContentLoaded', boot);
