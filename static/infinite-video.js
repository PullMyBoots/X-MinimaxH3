(() => {
  const q = (selector, root=document) => root.querySelector(selector);
  const qa = (selector, root=document) => [...root.querySelectorAll(selector)];
  const tr = value => window.H3I18n?.t(value) || value;
  const isEnglish = () => window.H3I18n?.locale === 'en';
  const activeStatuses = new Set(['queued', 'starting_backend', 'running', 'awaiting_preview']);
  const resolutionDetents = [360, 480, 540, 720, 900, 1080];
  const finalDetents = [720, 900, 1080, 1220, 1440];
  let projects = [];
  let project = null;
  let activeEngine = 'first_last';
  let activeOptions = null;
  let pollTimer = null;
  let previewObjectUrl = null;
  let finalObjectUrl = null;
  let promptDrafts = [];
  let promptDraftProjectId = null;
  let selectedWindowEntry = null;
  let retryDraft = null;
  let inputMode = 'manual';
  let projectCreationMode = 'online';
  let finalDefaultsProjectId = null;

  function isDirectJson(item=project) {
    return Boolean(item && item.workflow_version >= 3 && item.creation_mode === 'json');
  }

  function headers(json=false) {
    const value = {};
    const key = localStorage.getItem('h3serve_api_key');
    if (key) value['X-API-Key'] = key;
    if (json) value['Content-Type'] = 'application/json';
    return value;
  }

  async function request(path, init={}) {
    const response = await fetch(path, {...init, headers:{...headers(false), ...(init.headers || {})}});
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response;
  }

  function showMessage(selector, message, ok=false) {
    const node = q(selector);
    if (!node) return;
    node.textContent = message || '';
    node.hidden = !message;
    node.classList.toggle('success', Boolean(ok));
  }

  function escapeText(value) {
    const node = document.createElement('span');
    node.textContent = String(value ?? '');
    return node.innerHTML;
  }

  function secondsText(value, digits=2) {
    return isEnglish() ? `${Number(value).toFixed(digits)} s` : `${Number(value).toFixed(digits)} 秒`;
  }

  function bindRange(selector, outputSelector, formatter) {
    const input = q(selector);
    const output = q(outputSelector);
    if (!input) return;
    const update = () => { if (output) output.textContent = formatter(input.value); };
    input.addEventListener('input', update);
    update();
  }

  function bindFrameDropzone(selector) {
    const zone = q(selector);
    if (!zone) return;
    const input = q('input[type="file"]', zone);
    const image = q('img', zone);
    const remove = q('.remove-frame', zone);
    if (!input || !image || !remove) return;
    input.addEventListener('change', () => {
      if (!input.files?.[0]) return;
      image.src = URL.createObjectURL(input.files[0]);
      zone.classList.add('has-image');
    });
    remove.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      input.value = '';
      image.removeAttribute('src');
      zone.classList.remove('has-image');
    });
  }

  function clearWindowUploads() {
    const form = q('#infiniteWindowForm');
    if (!form) return;
    for (const input of form.querySelectorAll('input[type="file"]')) input.value = '';
    for (const zone of form.querySelectorAll('.dropzone')) {
      zone.classList.remove('has-image');
      q('img', zone)?.removeAttribute('src');
    }
    renderInfiniteReferencePreviews('image');
    renderInfiniteReferencePreviews('audio');
  }

  function infiniteReferenceInput(kind) {
    return q(kind === 'image' ? '#infiniteReferenceImages' : '#infiniteReferenceAudios');
  }

  function infiniteReferenceFiles(kind) {
    return Array.from(infiniteReferenceInput(kind)?.files || []).slice(0, kind === 'image' ? 9 : 3);
  }

  function renderInfiniteReferencePreviews(kind) {
    const preview = q(kind === 'image' ? '#infiniteReferencePreview' : '#infiniteReferenceAudioPreview');
    if (!preview) return;
    preview.innerHTML = infiniteReferenceFiles(kind).map((file, index) => {
      const label = kind === 'image' ? `Picture ${index + 1}` : `Audio ${index + 1}`;
      const visual = kind === 'image'
        ? `<img src="${URL.createObjectURL(file)}" alt="${escapeText(`${tr('参考图片')} ${index + 1}`)}">`
        : '<span class="reference-audio-icon">♫</span>';
      return `<article class="reference-chip"><button type="button" class="reference-chip-remove" data-infinite-reference-remove="${kind}:${index}" aria-label="${tr('删除素材')}">×</button><div class="reference-media-visual">${visual}</div><div class="reference-chip-copy"><button type="button" class="reference-token" data-infinite-reference-token="${kind}:${index}" title="${tr('插入当前提示词')}">&lt;${label}&gt;</button><small title="${escapeText(file.name)}">${escapeText(file.name)}</small></div></article>`;
    }).join('');
    qa('[data-infinite-reference-remove]', preview).forEach(button => button.addEventListener('click', () => {
      const [removeKind, rawIndex] = button.dataset.infiniteReferenceRemove.split(':');
      removeInfiniteReferenceFile(removeKind, Number(rawIndex));
    }));
    qa('[data-infinite-reference-token]', preview).forEach(button => button.addEventListener('click', () => {
      const [tokenKind, rawIndex] = button.dataset.infiniteReferenceToken.split(':');
      const token = `<${tokenKind === 'image' ? 'Picture' : 'Audio'} ${Number(rawIndex) + 1}>`;
      const target = q('#infiniteWindowForm [name="window_description"]');
      const start = target.selectionStart ?? target.value.length;
      const end = target.selectionEnd ?? start;
      const before = target.value.slice(0, start);
      const spacer = before && !/\s$/.test(before) ? ' ' : '';
      target.value = `${before}${spacer}${token} ${target.value.slice(end)}`;
      target.dispatchEvent(new Event('input', {bubbles:true}));
      target.focus();
    }));
  }

  function appendInfiniteReferenceFiles(kind, files) {
    const input = infiniteReferenceInput(kind);
    const maximum = kind === 'image' ? 9 : 3;
    const transfer = new DataTransfer();
    const existing = Array.from(input.files || []);
    const identities = new Set(existing.map(file => `${file.name}:${file.size}:${file.lastModified}`));
    const appended = Array.from(files || []).filter(file => {
      const identity = `${file.name}:${file.size}:${file.lastModified}`;
      if (identities.has(identity)) return false;
      identities.add(identity);
      return true;
    });
    [...existing, ...appended].slice(0, maximum).forEach(file => transfer.items.add(file));
    input.files = transfer.files;
    renderInfiniteReferencePreviews(kind);
  }

  function distributeInfiniteReferenceFiles(files) {
    const all = Array.from(files || []);
    appendInfiniteReferenceFiles('image', all.filter(file => file.type.startsWith('image/')));
    appendInfiniteReferenceFiles('audio', all.filter(file => file.type.startsWith('audio/') || /\.(wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name)));
    q('#infiniteReferenceFiles').value = '';
  }

  function removeInfiniteReferenceFile(kind, removeIndex) {
    const input = infiniteReferenceInput(kind);
    const transfer = new DataTransfer();
    Array.from(input.files || []).forEach((file, index) => {
      if (index !== removeIndex) transfer.items.add(file);
    });
    input.files = transfer.files;
    renderInfiniteReferencePreviews(kind);
  }

  function visualMemoryText(value) {
    return Number(value) ? (isEnglish() ? `${value} anchors` : `${value} 个锚点`) : tr('关闭');
  }

  function audioMemoryText(value) {
    return Number(value) ? (isEnglish() ? `${value} clips` : `${value} 段`) : tr('关闭');
  }

  function nearestDetent(raw, detents, threshold=10) {
    const value = Math.round(Number(raw));
    const nearest = detents.reduce((best, item) => Math.abs(item - value) < Math.abs(best - value) ? item : best, detents[0]);
    return Math.abs(nearest - value) <= threshold ? nearest : value;
  }

  function resolutionNumber(raw) {
    return String(raw || '').trim().toLowerCase() === '2k'
      ? 1440
      : Number.parseInt(raw, 10);
  }

  function geometryLabel(resolution, ratio) {
    const shortEdge = Number.parseInt(resolution, 10);
    const cached = activeOptions?.geometry?.[`${shortEdge}p`]?.[ratio];
    if (cached) return `${shortEdge}P · ${cached.width}×${cached.height}`;
    const [rw, rh] = String(ratio || '16:9').split(':').map(Number);
    const align32 = number => Math.max(32, Math.floor(number / 32 + 0.5) * 32);
    const width = rw >= rh ? align32(shortEdge * rw / rh) : align32(shortEdge);
    const height = rw >= rh ? align32(shortEdge) : align32(shortEdge * rh / rw);
    return `${shortEdge}P · ${width}×${height}`;
  }

  function setProjectResolution(raw, force=false) {
    const slider = q('#infiniteProjectResolutionSlider');
    if (!slider) return;
    const finalValue = Number.parseInt(q('#infiniteProjectForm')?.elements.final_resolution.value || '1080p', 10) || 1080;
    const nativeMaximum = Math.min(
      1080,
      Number(activeOptions?.resolution?.max) || 1080,
      finalValue,
    );
    let value = Math.max(Number(slider.min), Math.min(nativeMaximum, Math.round(Number.parseFloat(raw))));
    if (force) value = resolutionDetents.reduce((best, item) => Math.abs(item - value) < Math.abs(best - value) ? item : best, resolutionDetents[0]);
    else value = nearestDetent(value, resolutionDetents);
    value = Math.min(value, nativeMaximum);
    slider.value = String(value);
    const form = q('#infiniteProjectForm');
    form.elements.preview_resolution.value = `${value}p`;
    updateProjectResolutionTrack();
  }

  function configureProjectResolution() {
    const firstSlider = q('#infiniteProjectResolutionSlider');
    const finalSlider = q('#infiniteProjectFinalResolutionSlider');
    const form = q('#infiniteProjectForm');
    if (!firstSlider || !finalSlider || !form) return;
    const levels = activeOptions?.advanced_limits?.second_sampling?.levels || finalDetents.map(value => `${value}p`);
    const supportedFinalEdges = levels.map(resolutionNumber).filter(Number.isFinite);
    const finalMaximum = Math.max(...supportedFinalEdges, 1080);
    const minimum = Number(activeOptions?.resolution?.min) || 360;
    firstSlider.min = finalSlider.min = String(minimum);
    firstSlider.max = finalSlider.max = String(finalMaximum);
    const requestedFinal = Number(form.elements.final_resolution.value) || Number(finalSlider.value) || 1080;
    const finalValue = Math.max(minimum, Math.min(finalMaximum, Math.round(requestedFinal)));
    finalSlider.value = String(finalValue);
    form.elements.final_resolution.value = `${finalValue}p`;
    const firstMaximum = Math.min(1080, Number(activeOptions?.resolution?.max) || 1080, finalValue);
    const requestedFirst = Number(form.elements.preview_resolution.value) || Number(firstSlider.value) || 540;
    const firstValue = Math.max(minimum, Math.min(firstMaximum, requestedFirst));
    firstSlider.value = String(firstValue);
    form.elements.preview_resolution.value = `${firstValue}p`;
    updateProjectResolutionTrack();
  }

  function setProjectFinalResolution(raw, force=false) {
    const slider = q('#infiniteProjectFinalResolutionSlider');
    if (!slider) return;
    const firstValue = Number.parseInt(q('#infiniteProjectForm')?.elements.preview_resolution.value || '540p', 10) || 540;
    const availableDetents = finalDetents.filter(item => (
      item >= firstValue
      && item >= Number(slider.min)
      && item <= Number(slider.max)
    ));
    let value = Math.max(firstValue, Math.min(Number(slider.max), Math.round(Number.parseFloat(raw))));
    if (availableDetents.length) {
      const nearest = availableDetents.reduce(
        (best, item) => Math.abs(item - value) < Math.abs(best - value) ? item : best,
        availableDetents[0],
      );
      if (force || Math.abs(nearest - value) <= 10) value = nearest;
    }
    slider.value = String(value);
    q('#infiniteProjectForm').elements.final_resolution.value = `${value}p`;
    updateProjectResolutionTrack();
  }

  function configureProjectFinalResolution() {
    configureProjectResolution();
  }

  function updateProjectResolutionTrack() {
    const form = q('#infiniteProjectForm');
    const firstSlider = q('#infiniteProjectResolutionSlider');
    const finalSlider = q('#infiniteProjectFinalResolutionSlider');
    const track = q('#infiniteProjectResolutionTrack');
    if (!form || !firstSlider || !finalSlider || !track) return;
    const first = Number.parseInt(form.elements.preview_resolution.value, 10) || 540;
    const final = Number.parseInt(form.elements.final_resolution.value, 10) || 1080;
    const minimum = Number(finalSlider.min) || 360;
    const maximum = Math.max(minimum + 1, Number(finalSlider.max) || 1440);
    const percentage = value => Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum))) * 100;
    track.style.setProperty('--first-step-percent', `${percentage(first)}%`);
    track.style.setProperty('--total-step-percent', `${percentage(final)}%`);
    q('#infiniteProjectFirstResolutionValue').textContent = `${first}P`;
    q('#infiniteProjectFinalResolutionValue').textContent = `${final}P`;
    q('#infiniteProjectResolutionValue').textContent = first === final
      ? `${final}P · ${tr('全程同分辨率')}`
      : `${isEnglish() ? 'First' : '一采'} ${first}P · ${isEnglish() ? 'Final' : '二采'} ${final}P`;
    q('#infiniteProjectResolutionHint').textContent = first === final
      ? tr('两个节点重合时全程使用同一分辨率，不执行 latent 放大。')
      : tr('绿色节点不能超过橙色节点；完成一次采样后执行 latent 放大。');
    qa('[data-infinite-final-resolution]').forEach(button => {
      const point = Number.parseInt(button.dataset.infiniteFinalResolution, 10);
      const disabled = point < first || point < minimum || point > maximum;
      button.disabled = disabled;
      button.classList.toggle('active', !disabled && point === final);
      button.setAttribute('aria-pressed', String(!disabled && point === final));
    });
  }

  function updateProjectSamplingTrack(resetForVariant=false) {
    const form = q('#infiniteProjectForm');
    const total = form?.elements.sampling_steps;
    const first = q('#infiniteProjectFirstPassSteps');
    const final = form?.elements.final_sampling_steps;
    const track = q('#infiniteProjectStepTrack');
    if (!total || !first || !final || !track) return;
    const lora = form.elements.model_variant.value === 'lora';
    const maximum = lora ? 10 : 30;
    total.min = '1';
    total.max = String(maximum);
    first.min = '1';
    first.max = String(maximum);
    if (resetForVariant) {
      total.value = lora ? '8' : '20';
      first.value = lora ? '6' : '18';
    }
    const totalSteps = Math.max(4, Math.min(maximum, Number(total.value) || (lora ? 8 : 20)));
    total.value = String(totalSteps);
    const minimumFirst = Math.max(1, totalSteps - 8);
    const firstSteps = Math.max(minimumFirst, Math.min(totalSteps - 1, Number(first.value) || totalSteps - 2));
    first.value = String(firstSteps);
    const secondSteps = totalSteps - firstSteps;
    final.value = String(secondSteps);
    const percent = value => maximum <= 1 ? 100 : ((value - 1) / (maximum - 1)) * 100;
    track.style.setProperty('--first-step-percent', `${percent(firstSteps)}%`);
    track.style.setProperty('--total-step-percent', `${percent(totalSteps)}%`);
    q('#infiniteProjectFirstPassStepsValue').textContent = isEnglish() ? `${firstSteps} steps` : `${firstSteps} 步`;
    q('#infiniteProjectTotalStepsValue').textContent = isEnglish() ? `${totalSteps} steps` : `${totalSteps} 步`;
    q('#infiniteProjectSamplingStepsValue').textContent = isEnglish()
      ? `First ${firstSteps} · second ${secondSteps} · ${totalSteps} total`
      : `一采 ${firstSteps} · 二采 ${secondSteps} · 总计 ${totalSteps} 步`;
  }

  function syncProjectModelVariantToggle() {
    const form = q('#infiniteProjectForm');
    const toggle = q('#infiniteProjectLoraEnabled');
    const summary = q('#infiniteProjectModelVariantSummary');
    if (!form || !toggle || !summary) return;
    const lora = form.elements.model_variant.value === 'lora';
    toggle.checked = lora;
    summary.textContent = tr(lora ? '开启 · LoRA 高速' : '关闭 · INT8 原始权重');
  }

  function configureFinalResolution() {
    if (!project) return false;
    const form = q('#infiniteFinalSamplingForm');
    form.elements.resolution.value = String(
      project.final_sampling?.settings?.resolution || project.final_resolution || '1080p'
    );
    form.elements.steps.value = String(
      project.final_sampling?.settings?.steps ?? project.final_sampling_steps ?? 4
    );
    return true;
  }

  function projectSettingsText(item) {
    const variant = item.model_variant === 'lora' ? 'LoRA' : 'Base';
    const resolution = String(item.preview_resolution || item.resolution || '').toUpperCase();
    if (item.workflow_version >= 3) {
      const finalResolution = String(item.final_resolution || '1080p').toUpperCase();
      const lowSteps = Math.max(1, Number(item.sampling_steps) - Number(item.final_sampling_steps));
      const preview = item.creation_mode === 'online' && item.preview_enabled
        ? (isEnglish() ? ` · preview +${item.preview_branch_steps || 2}` : ` · 预览补 ${item.preview_branch_steps || 2} 步`)
        : '';
      if (isEnglish()) return `${resolution} → ${finalResolution} · ${item.aspect_ratio} · ${variant} ${item.sampling_steps} total (${lowSteps} low + ${item.final_sampling_steps} high)${preview} · acceleration is set per window and at final generation`;
      return `${resolution} → ${finalResolution} · ${item.aspect_ratio} · ${variant} 总${item.sampling_steps}步（一采${lowSteps} + 二采${item.final_sampling_steps}）${preview} · 加速按窗口与最终生成分别设置`;
    }
    if (isEnglish()) return `${resolution} preview · ${item.aspect_ratio} · ${variant} ${item.sampling_steps} steps · acceleration ${item.acceleration} · ${secondsText(item.window_duration_seconds, 1)}/window`;
    return `${resolution} 预览 · ${item.aspect_ratio} · ${variant} ${item.sampling_steps} 步 · 加速 ${item.acceleration} · 每窗 ${secondsText(item.window_duration_seconds, 1)}`;
  }

  function renderProjectList() {
    const root = q('#infiniteProjectList');
    if (!projects.length) {
      root.innerHTML = `<div class="infinite-project-empty">${tr('还没有长视频项目')}</div>`;
      return;
    }
    root.innerHTML = projects.map(item => `<button type="button" class="infinite-project-button ${project?.id === item.id ? 'active' : ''}" data-infinite-project="${item.id}"><strong>${escapeText(item.title)}</strong><small>${isEnglish() ? `${item.window_count} windows` : `${item.window_count} 个窗口`} · ${secondsText(item.duration_seconds || 0)}</small></button>`).join('');
    qa('[data-infinite-project]', root).forEach(button => button.addEventListener('click', () => selectProject(button.dataset.infiniteProject)));
  }

  function draftStorageKey(projectId) {
    return `h3serve_infinite_prompt_drafts:${projectId}`;
  }

  function inheritedWindowValues(source=null) {
    const item = source || project?.windows?.at(-1) || {};
    return {
      window_description: '',
      duration_seconds: Number(item.requested_duration_seconds ?? project?.window_duration_seconds ?? 5),
      overlap_seconds: Number(item.overlap_seconds ?? project?.overlap_seconds ?? 1.625),
      acceleration: Number(item.acceleration ?? project?.acceleration ?? 50),
      visual_memory_capacity: Number(item.visual_memory_capacity ?? project?.visual_memory_capacity ?? 6),
      audio_memory_capacity: Number(item.audio_memory_capacity ?? project?.audio_memory_capacity ?? 1),
      visual_memory_resolution: String(item.visual_memory_resolution ?? project?.visual_memory_resolution ?? '360p'),
      memory_enabled: Boolean(Number(item.visual_memory_capacity ?? project?.visual_memory_capacity ?? 6) || Number(item.audio_memory_capacity ?? project?.audio_memory_capacity ?? 1)),
      inherit_references: true,
      excluded_reference_roles: [],
      seed: String(item.seed ?? 'random'),
    };
  }

  function normalizeDraft(value) {
    if (typeof value === 'string') return {...inheritedWindowValues(), window_description:value};
    const normalized = {...inheritedWindowValues(), ...(value && typeof value === 'object' ? value : {})};
    normalized.excluded_reference_roles = Array.isArray(normalized.excluded_reference_roles)
      ? normalized.excluded_reference_roles.map(String)
      : [];
    return normalized;
  }

  function loadPromptDrafts() {
    if (!project || promptDraftProjectId === project.id) return;
    promptDraftProjectId = project.id;
    selectedWindowEntry = null;
    retryDraft = null;
    try {
      const stored = JSON.parse(localStorage.getItem(draftStorageKey(project.id)) || '{}');
      promptDrafts = Array.isArray(stored.drafts) ? stored.drafts.map(normalizeDraft) : [];
      retryDraft = stored.retry_draft ? normalizeDraft(stored.retry_draft) : (stored.retry_prompt ? normalizeDraft(stored.retry_prompt) : null);
    } catch (_) {
      promptDrafts = [];
    }
    if (!promptDrafts.length && !project.windows.length) promptDrafts = [inheritedWindowValues()];
  }

  function persistPromptDrafts() {
    if (!project) return;
    localStorage.setItem(draftStorageKey(project.id), JSON.stringify({drafts:promptDrafts, retry_draft:retryDraft}));
  }

  function readWindowForm() {
    const form = q('#infiniteWindowForm');
    const selectedValue = selectedWindowEntry?.kind === 'draft'
      ? promptDrafts[selectedWindowEntry.index]
      : selectedWindowEntry?.kind === 'retry' ? retryDraft : null;
    return {
      window_description: form.elements.window_description.value,
      duration_seconds: Number(form.elements.duration_seconds.value),
      overlap_seconds: Number(form.elements.overlap_seconds.value),
      acceleration: Number(form.elements.acceleration.value),
      visual_memory_capacity: q('#infiniteMemoryEnabled').checked ? Number(form.elements.visual_memory_capacity.value) : 0,
      audio_memory_capacity: q('#infiniteMemoryEnabled').checked ? Number(form.elements.audio_memory_capacity.value) : 0,
      visual_memory_resolution: form.elements.visual_memory_resolution.value,
      memory_enabled: q('#infiniteMemoryEnabled').checked,
      inherit_references: Boolean(form.elements.inherit_references?.checked),
      excluded_reference_roles: [...(selectedValue?.excluded_reference_roles || [])],
      seed: form.elements.seed.value || 'random',
    };
  }

  function updateWindowOutputs() {
    const form = q('#infiniteWindowForm');
    const overlap = Number(form.elements.overlap_seconds.value) || 0;
    const duration = form.elements.duration_seconds;
    const maximum = Math.max(1, 15 - overlap);
    duration.max = String(maximum);
    if (Number(duration.value) > maximum) duration.value = String(Math.floor(maximum * 4) / 4);
    q('#infiniteWindowDurationValue').textContent = secondsText(form.elements.duration_seconds.value, 1);
    q('#infiniteWindowOverlapValue').textContent = secondsText(form.elements.overlap_seconds.value, 3);
    q('#infiniteWindowAccelerationValue').textContent = form.elements.acceleration.value;
    const enabled = q('#infiniteMemoryEnabled').checked;
    q('#infiniteMemoryControls').hidden = !enabled;
    q('#infiniteVisualMemoryValue').textContent = form.elements.visual_memory_capacity.value;
    q('#infiniteAudioMemoryValue').textContent = form.elements.audio_memory_capacity.value;
    q('#infiniteMemorySummary').textContent = enabled
      ? `${tr('开启')} · ${tr('视觉')} ${form.elements.visual_memory_capacity.value} · ${tr('音频')} ${form.elements.audio_memory_capacity.value}`
      : tr('关闭');
  }

  function writeWindowForm(value) {
    const form = q('#infiniteWindowForm');
    const item = normalizeDraft(value);
    clearWindowUploads();
    for (const name of ['window_description', 'duration_seconds', 'overlap_seconds', 'acceleration', 'visual_memory_capacity', 'audio_memory_capacity', 'visual_memory_resolution', 'seed']) {
      if (form.elements[name]) form.elements[name].value = item[name];
    }
    q('#infiniteMemoryEnabled').checked = item.memory_enabled !== false;
    if (form.elements.inherit_references) form.elements.inherit_references.checked = item.inherit_references !== false;
    updateWindowOutputs();
  }

  function saveSelectedPrompt() {
    if (!selectedWindowEntry) return;
    const form = q('#infiniteWindowForm');
    if (!form || form.elements.window_description.readOnly) return;
    const value = readWindowForm();
    if (selectedWindowEntry.kind === 'draft' && promptDrafts[selectedWindowEntry.index] !== undefined) promptDrafts[selectedWindowEntry.index] = value;
    if (selectedWindowEntry.kind === 'retry') retryDraft = value;
    persistPromptDrafts();
  }

  function promptSummary(value) {
    return String(value || '').replace(/\s+/g, ' ').trim() || tr('点击进入并填写这个窗口的提示词');
  }

  function statusText(value) {
    if (isDirectJson()) {
      const labels = isEnglish()
        ? {queued:'Queued', starting_backend:'Preparing', running:'Generating final window', succeeded:'Final window complete', failed:'Failed', cancelled:'Cancelled', missing:'Missing'}
        : {queued:'排队中', starting_backend:'准备中', running:'生成正式窗口中', succeeded:'正式窗口已完成', failed:'失败', cancelled:'已取消', missing:'记录缺失'};
      return labels[value] || value;
    }
    const labels = isEnglish()
      ? {queued:'Queued', starting_backend:'Preparing', running:'Generating preview', succeeded:'Preview ready', failed:'Failed', cancelled:'Cancelled', missing:'Missing'}
      : {queued:'排队中', starting_backend:'准备中', running:'生成预览中', succeeded:'预览已完成', failed:'失败', cancelled:'已取消', missing:'记录缺失'};
    return labels[value] || value;
  }

  function windowProgress(item) {
    const reported = Number(item?.progress?.percent);
    if (item?.status === 'succeeded') return 100;
    if (!activeStatuses.has(item?.status)) return 0;
    if (!Number.isFinite(reported)) return 0;
    return Math.max(0, Math.min(100, reported));
  }

  function windowProgressLabel(item, percent) {
    const rounded = Math.round(percent);
    if (item.status === 'succeeded') return isEnglish() ? 'Generated · 100%' : '已生成 · 100%';
    if (activeStatuses.has(item.status)) return `${statusText(item.status)} · ${rounded}%`;
    return `${statusText(item.status)} · ${rounded}%`;
  }

  function renderWindowList() {
    if (!project) return;
    loadPromptDrafts();
    const root = q('#infiniteWindowList');
    const submitted = project.windows.map((item, index) => {
      const retryable = project.can_retry_tail && index === project.windows.length - 1;
      const kind = retryable ? 'retry' : 'submitted';
      const active = selectedWindowEntry?.kind === kind && selectedWindowEntry.index === index;
      const description = retryable && retryDraft?.window_description ? retryDraft.window_description : item.window_description;
      const removable = index === project.windows.length - 1 && promptDrafts.length === 0 && !activeStatuses.has(item.status);
      const percent = windowProgress(item);
      const stateLabel = retryable ? `${tr('可修改后重试')} · ${Math.round(percent)}%` : windowProgressLabel(item, percent);
      return `<article class="infinite-window-row"><button type="button" class="infinite-window-select ${item.status} ${percent > 0 ? 'has-progress' : ''} ${active ? 'active' : ''}" style="--window-progress:${percent}%" data-window-kind="${kind}" data-window-index="${index}" aria-label="${escapeText(`${isEnglish() ? 'Window' : '窗口'} ${index + 1} · ${stateLabel}`)}"><i class="infinite-window-progress" aria-hidden="true"></i><span><strong>${isEnglish() ? 'Window' : '窗口'} ${index + 1}</strong><small>${stateLabel}</small></span><span>${escapeText(promptSummary(description))}</span></button>${removable ? `<button type="button" class="infinite-window-remove" data-remove-submitted-tail="true" aria-label="${tr('删除尾部窗口')}">×</button>` : ''}</article>`;
    });
    const batchRunning = project.batch?.status === 'running';
    const drafts = batchRunning ? [] : promptDrafts.map((draft, index) => {
      const active = selectedWindowEntry?.kind === 'draft' && selectedWindowEntry.index === index;
      const number = project.windows.length + index + 1;
      const removable = index === promptDrafts.length - 1;
      const stateLabel = `${index === 0 ? tr('下一个可提交') : tr('未提交草稿')} · 0%`;
      return `<article class="infinite-window-row"><button type="button" class="infinite-window-select draft ${active ? 'active' : ''}" style="--window-progress:0%" data-window-kind="draft" data-window-index="${index}"><i class="infinite-window-progress" aria-hidden="true"></i><span><strong>${isEnglish() ? 'Window' : '窗口'} ${number}</strong><small>${stateLabel}</small></span><span>${escapeText(promptSummary(draft.window_description))}</span></button>${removable ? `<button type="button" class="infinite-window-remove" data-remove-window-draft="${index}">×</button>` : ''}</article>`;
    });
    root.innerHTML = [...submitted, ...drafts].join('') || `<div class="infinite-window-list-empty">${tr('还没有窗口提示词，点击“添加窗口”开始。')}</div>`;
    qa('[data-window-kind]', root).forEach(button => button.addEventListener('click', () => selectWindowEntry(button.dataset.windowKind, Number(button.dataset.windowIndex))));
    qa('[data-remove-window-draft]', root).forEach(button => button.addEventListener('click', () => removeWindowDraft(Number(button.dataset.removeWindowDraft))));
    qa('[data-remove-submitted-tail]', root).forEach(button => button.addEventListener('click', deleteSubmittedTail));
    renderWindowEditor();
  }

  function renderWindowEditor() {
    const form = q('#infiniteWindowForm');
    if (!project || !selectedWindowEntry || inputMode !== 'manual') {
      form.hidden = true;
      return;
    }
    const editable = ['draft', 'retry'].includes(selectedWindowEntry.kind);
    const number = selectedWindowEntry.kind === 'draft' ? project.windows.length + selectedWindowEntry.index + 1 : selectedWindowEntry.index + 1;
    form.hidden = false;
    form.classList.toggle('read-only-window', !editable);
    for (const control of form.querySelectorAll('textarea,input:not([type="hidden"]),select')) control.disabled = !editable;
    form.elements.window_description.readOnly = !editable;
    q('#infiniteWindowTitle').textContent = isEnglish() ? `Edit window ${number}` : `编辑第 ${number} 个窗口`;
    q('#infiniteInheritReferencesRow').hidden = project.window_count === 0;
    const inherited = project.windows.at(-1)?.conditioning || {};
    const inheritedParts = [];
    if (inherited.has_first_frame) inheritedParts.push(tr('首帧'));
    if (inherited.has_last_frame) inheritedParts.push(tr('尾帧'));
    if (inherited.reference_image_count) inheritedParts.push(`${inherited.reference_image_count} ${tr('张参考图')}`);
    if (inherited.reference_audio_count) inheritedParts.push(`${inherited.reference_audio_count} ${tr('段参考音频')}`);
    q('#infiniteReferenceInheritanceSummary').textContent = project.window_count
      ? (inheritedParts.length ? `${tr('可继承')}：${inheritedParts.join(' · ')}` : tr('上一窗口没有参考素材'))
      : tr('首窗口尚无可继承素材');
    const referenceList = q('#infiniteInheritedReferenceList');
    const selectedValue = selectedWindowEntry.kind === 'draft'
      ? promptDrafts[selectedWindowEntry.index]
      : selectedWindowEntry.kind === 'retry' ? retryDraft : null;
    const excluded = new Set(selectedValue?.excluded_reference_roles || []);
    const roles = Array.isArray(inherited.roles) ? inherited.roles : [];
    referenceList.hidden = !project.window_count || !roles.length || !editable || !form.elements.inherit_references?.checked;
    referenceList.innerHTML = roles.map(item => {
      const removed = excluded.has(item.role);
      const index = item.role.split('_').at(-1);
      const label = item.role === 'first_frame' ? tr('首帧')
        : item.role === 'last_frame' ? tr('尾帧')
        : item.role.startsWith('reference_image_')
          ? (isEnglish() ? `Reference image ${index}` : `参考图 ${index}`)
          : (isEnglish() ? `Reference audio ${index}` : `参考音频 ${index}`);
      return `<button type="button" class="${removed ? 'excluded' : ''}" data-inherited-role="${escapeText(item.role)}">${escapeText(label)} ${removed ? '↶' : '×'}</button>`;
    }).join('');
    qa('[data-inherited-role]', referenceList).forEach(button => button.addEventListener('click', () => {
      const role = button.dataset.inheritedRole;
      const current = new Set(selectedValue?.excluded_reference_roles || []);
      if (current.has(role)) current.delete(role); else current.add(role);
      selectedValue.excluded_reference_roles = [...current];
      persistPromptDrafts();
      renderWindowEditor();
    }));
    const nextDraft = selectedWindowEntry.kind === 'draft' && selectedWindowEntry.index === 0;
    const retry = selectedWindowEntry.kind === 'retry' && project.can_retry_tail;
    const allowed = editable && project.can_append && project.batch?.status !== 'running' && (nextDraft || retry);
    q('#appendInfiniteWindow').disabled = !allowed;
    q('#appendInfiniteWindow').textContent = selectedWindowEntry.kind === 'submitted' ? tr('这个窗口已经生成') : retry ? tr('重新生成并替换失败尾段') : nextDraft ? tr('生成预览并接到尾部') : tr('请先提交前面的窗口');
  }

  function selectWindowEntry(kind, index) {
    saveSelectedPrompt();
    selectedWindowEntry = {kind, index};
    let value;
    if (kind === 'draft') value = promptDrafts[index];
    else {
      const item = project.windows[index];
      if (kind === 'retry' && !retryDraft) retryDraft = normalizeDraft(item);
      value = kind === 'retry' ? retryDraft : item;
    }
    writeWindowForm(value);
    renderWindowList();
    q('#infiniteWindowForm').scrollIntoView({behavior:'smooth', block:'start'});
  }

  function addWindowDraft() {
    saveSelectedPrompt();
    promptDrafts.push(inheritedWindowValues());
    persistPromptDrafts();
    selectWindowEntry('draft', promptDrafts.length - 1);
  }

  function removeWindowDraft(index) {
    saveSelectedPrompt();
    if (promptDrafts[index] === undefined || index !== promptDrafts.length - 1) return;
    promptDrafts.splice(index, 1);
    selectedWindowEntry = null;
    persistPromptDrafts();
    renderWindowList();
  }

  function jsonTemplate() {
    const windows = [
      {prompt:'[Shot 1] Describe window 1 from local 00:00 to the exact ending state.', seed:'random', acceleration:50},
      {prompt:'[Shot 1] Continue the exact ending state without replay. Describe window 2.', seed:'random', acceleration:50}
    ];
    if (activeEngine === 'reference') {
      windows[0].references = {
        'Picture 1': {'path':'/absolute/path/to/window-1-scene.png'},
        'Picture 2': {'path':'/absolute/path/to/window-1-character.png'}
      };
      windows[0].prompt = '[Shot 1] Use <Picture 1> as the scene reference and <Picture 2> as the character reference. Describe window 1 from local 00:00 to the exact ending state.';
      windows[1].references = {
        'Picture 1': {'path':'/absolute/path/to/window-2-scene.png'},
        'Picture 2': {'path':'/absolute/path/to/window-2-character.png'}
      };
      windows[1].prompt = '[Shot 1] Continue the exact ending state without replay. Use this window\'s <Picture 1> and <Picture 2> references, then describe the next action and exact ending state.';
    }
    return JSON.stringify({
      final_acceleration: project?.second_pass_acceleration ?? 75,
      windows
    }, null, 2);
  }

  function setProjectCreationMode(mode) {
    projectCreationMode = mode === 'json' ? 'json' : 'online';
    const form = q('#infiniteProjectForm');
    form.elements.creation_mode.value = projectCreationMode;
    qa('[data-project-mode]').forEach(button => button.classList.toggle('active', button.dataset.projectMode === projectCreationMode));
    q('#infiniteJsonFoundation').hidden = projectCreationMode !== 'json';
    q('#createInfiniteProject').textContent = projectCreationMode === 'json' ? tr('导入并开始生成 →') : tr('进入在线创作 →');
    if (projectCreationMode === 'json' && !q('#infiniteProjectJsonInput').value.trim()) q('#infiniteProjectJsonInput').value = jsonTemplate();
  }

  function switchInputMode() {
    inputMode = project?.creation_mode === 'json' ? 'json' : 'manual';
    q('#infiniteManualMode').hidden = inputMode !== 'manual';
    q('#infiniteJsonMode').hidden = inputMode !== 'json';
    q('#infiniteCreationModeBadge').textContent = inputMode === 'json' ? tr('JSON 一键创作') : tr('在线创作');
    q('#infiniteCreationModeDescription').textContent = inputMode === 'json'
      ? tr('后台按脚本顺序逐窗运行完整 SelfLift 正式轨迹，完成后直接输出最终高清成片。')
      : tr('逐窗口编辑、生成和审阅，下一窗口继承上一窗口设置。');
    if (inputMode === 'json' && !q('#infiniteJsonInput').value.trim()) q('#infiniteJsonInput').value = jsonTemplate();
    renderWindowEditor();
  }

  async function loadVideo(videoUrl, kind='preview') {
    try {
      const response = await request(videoUrl);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      if (kind === 'final') {
        if (finalObjectUrl) URL.revokeObjectURL(finalObjectUrl);
        finalObjectUrl = url;
        q('#infiniteFinalVideo').src = url;
        q('#downloadInfiniteFinal').href = url;
        q('#infiniteFinalResult').hidden = false;
      } else {
        if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = url;
        q('#infiniteVideo').src = url;
        q('#infiniteVideo').dataset.source = videoUrl;
        q('.infinite-video-stage').classList.add('has-video');
        q('#downloadInfiniteVideo').href = url;
        q('#downloadInfiniteVideo').hidden = false;
      }
    } catch (error) {
      showMessage(kind === 'final' ? '#infiniteFinalMessage' : '#infiniteWindowMessage', error.message);
    }
  }

  function renderBatchState() {
    const batch = project.batch || {status:'idle', cursor:0, total:0};
    const running = batch.status === 'running';
    const directJson = isDirectJson();
    q('#startInfiniteBatch').disabled = running || !project.trajectory_locked || Boolean(project.final_sampling?.active);
    q('#stopInfiniteBatch').hidden = !running;
    q('#addInfiniteWindowPrompt').disabled = running;
    const message = running
      ? (directJson
        ? (isEnglish() ? `Final generation: ${batch.cursor}/${batch.total} windows submitted.` : `正式成片生成：已提交 ${batch.cursor} / ${batch.total} 个窗口。`)
        : (isEnglish() ? `Backend queue: ${batch.cursor}/${batch.total} windows submitted.` : `后台自动队列：已提交 ${batch.cursor} / ${batch.total} 个窗口。`))
      : batch.status === 'completed'
        ? (directJson
          ? (isEnglish() ? `All ${batch.total} formal windows are complete. The final high-resolution film is ready.` : `${batch.total} 个正式窗口已全部完成，最终高清成片已生成。`)
          : (isEnglish() ? `All ${batch.total} preview windows are complete. Second sampling has not started.` : `${batch.total} 个预览窗口已全部完成，尚未启动整片二次采样。`))
        : batch.status === 'failed'
          ? `${tr('自动队列已停止：')}${batch.error || tr('窗口生成失败')}`
          : batch.status === 'stopped' ? tr('自动队列已停止。') : '';
    showMessage('#infiniteJsonMessage', message, running || batch.status === 'completed');
  }

  function renderFinalSampling() {
    const directJson = isDirectJson();
    q('#infiniteFinalSampling').hidden = directJson;
    if (directJson) return;
    const targetAvailable = configureFinalResolution();
    const final = project.final_sampling || {};
    const finalActive = activeStatuses.has(final.status);
    const batchRunning = project.batch?.status === 'running';
    const ready = targetAvailable && Boolean(project.second_sampling_available) && !batchRunning && !activeStatuses.has(project.tail_status);
    const button = q('#submitInfiniteFinal');
    button.disabled = !ready || finalActive;
    button.textContent = finalActive ? tr('整片二次采样进行中…') : tr('对整片进行二次采样');
    const status = q('#infiniteFinalStatus');
    if (final.status === 'succeeded') status.textContent = tr('整片二次采样已完成');
    else if (finalActive) status.textContent = tr('整片二次采样进行中');
    else if (final.status === 'failed' || final.status === 'cancelled') status.textContent = tr('上次二次采样失败，可重新提交');
    else if (ready) status.textContent = tr('可以开始整片二次采样');
    else if (!targetAvailable && project.window_count) status.textContent = tr('一次采样分辨率已达到当前后端上限');
    else status.textContent = batchRunning ? tr('等待自动预览队列完成') : tr('等待至少一个完整预览窗口');
    q('#infiniteFinalSamplingForm').elements.acceleration.disabled = false;
    if (final.video_url) {
      if (q('#infiniteFinalVideo').dataset.source !== final.video_url) {
        q('#infiniteFinalVideo').dataset.source = final.video_url;
        loadVideo(final.video_url, 'final');
      }
    } else {
      q('#infiniteFinalResult').hidden = true;
      q('#infiniteFinalVideo').removeAttribute('src');
      q('#infiniteFinalVideo').dataset.source = '';
    }
  }

  function renderProject() {
    if (!project) return;
    loadPromptDrafts();
    q('#infiniteProjectForm').hidden = true;
    q('#infiniteWorkspace').hidden = false;
    q('#infiniteProjectTitle').textContent = project.title;
    const directJson = isDirectJson();
    const outputResolution = directJson ? project.final_resolution : (project.preview_resolution || project.resolution);
    const outputKind = directJson ? (isEnglish() ? 'final' : '正式成片') : 'preview';
    q('#infiniteProjectMeta').textContent = `${project.window_count} ${isEnglish() ? 'windows' : '个窗口'} · ${secondsText(project.duration_seconds || 0)} · ${String(outputResolution || '').toUpperCase()} ${outputKind}`;
    q('#infiniteOutputEyebrow').textContent = directJson ? 'FINAL OUTPUT' : 'CUMULATIVE PREVIEW';
    q('#downloadInfiniteVideo').textContent = directJson ? tr('下载最终成片') : tr('下载预览');
    q('#infiniteVideoEmpty').textContent = directJson
      ? tr('JSON 队列完成后，最终高清成片会出现在这里。')
      : tr('提交第一个窗口后，低分辨率累计预览会出现在这里。');
    const trajectoryTitle = directJson ? 'SelfLift direct trajectory' : project.workflow_version >= 3 ? 'SelfLift Y trajectory' : project.trajectory_locked ? tr('已锁定的预览轨迹') : tr('旧版项目参数');
    const trajectoryNote = project.trajectory_locked
      ? (project.workflow_version >= 3
        ? (directJson
          ? (isEnglish() ? 'Windows accumulate one connected low-resolution latent track without preview decoding. After the last window, H3 lifts the whole track once, completes the overlapping high-resolution tail, and decodes one final video.' : '各窗口只累积一条连续的低分辨率 latent，不做预览解码；最后一个窗口完成后，全片统一升维、重叠滑窗完成高清尾步，并只解码一次成片。')
          : (isEnglish() ? 'Each window publishes a low-resolution preview while retaining the same connected source latent. One-click final generation lifts and completes the whole approved track globally.' : '每个窗口快速发布低分辨率预览，同时保留同一条连续的一采 latent；一键定稿时再对整条已确认轨迹统一升维并完成高清尾步。'))
        : (isEnglish() ? 'Each window produces a cumulative preview and retains its native H3 latent. Full-film second sampling is manual.' : '每个窗口生成累计预览并保留原生 H3 latent；整片二次采样只由你手动启动。'))
      : (isEnglish() ? 'This legacy project keeps using the settings of its latest window. JSON automation is available in new projects.' : '这个旧版项目继续沿用上一窗口参数；JSON 自动生成只用于新建项目。');
    q('#infiniteProjectContract').innerHTML = `<strong>${trajectoryTitle}</strong><span>${escapeText(projectSettingsText(project))}</span><small>${trajectoryNote}</small>`;
    switchInputMode();
    if (finalDefaultsProjectId !== project.id) {
      finalDefaultsProjectId = project.id;
      const settings = project.final_sampling?.settings || {};
      q('#infiniteFinalSamplingForm').elements.resolution.value = String(settings.resolution || project.final_resolution || '1080p');
      q('#infiniteFinalSamplingForm').elements.steps.value = String(settings.steps ?? project.final_sampling_steps ?? 4);
      q('#infiniteFinalSamplingForm').elements.acceleration.value = String(settings.acceleration ?? project.second_pass_acceleration ?? 75);
      q('#infiniteFinalSamplingForm').elements.sigma_scale.value = String(settings.sigma_scale ?? 1);
      q('#infiniteFinalAccelerationValue').textContent = q('#infiniteFinalSamplingForm').elements.acceleration.value;
      q('#infiniteFinalSigmaScaleValue').textContent = Number(q('#infiniteFinalSamplingForm').elements.sigma_scale.value).toFixed(2);
    }
    renderWindowList();
    renderBatchState();
    renderFinalSampling();
    const accepted = project.windows.filter(item => item.status === 'succeeded').at(-1);
    const videoUrl = accepted?.video_url;
    if (videoUrl && q('#infiniteVideo').dataset.source !== videoUrl) loadVideo(videoUrl);
    if (!videoUrl) {
      q('#infiniteVideo').removeAttribute('src');
      q('#infiniteVideo').dataset.source = '';
      q('.infinite-video-stage').classList.remove('has-video');
      q('#downloadInfiniteVideo').hidden = true;
    }
    clearTimeout(pollTimer);
    const polling = project.batch?.status === 'running' || activeStatuses.has(project.tail_status) || activeStatuses.has(project.final_sampling?.status);
    if (polling && !q('#infinitePage').hidden) pollTimer = setTimeout(refreshSelected, 1500);
  }

  async function loadProjects(selectNewest=false) {
    const response = await request('/api/v1/infinite-projects');
    projects = (await response.json()).projects || [];
    if (selectNewest && projects.length) project = projects[0];
    else if (project) project = projects.find(item => item.id === project.id) || null;
    renderProjectList();
    if (project) renderProject();
  }

  async function selectProject(id) {
    saveSelectedPrompt();
    const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(id)}`);
    project = await response.json();
    renderProjectList();
    renderProject();
  }

  async function refreshSelected() {
    if (!project || q('#infinitePage').hidden) return;
    try {
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}`);
      project = await response.json();
      await loadProjects();
    } catch (error) {
      showMessage('#infiniteWindowMessage', error.message);
    }
  }

  async function createProject(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const body = Object.fromEntries(new FormData(form).entries());
    delete body.preview_branch_steps;
    body.preview_enabled = projectCreationMode === 'online';
    let script = null;
    try {
      if (projectCreationMode === 'json') {
        script = JSON.parse(q('#infiniteProjectJsonInput').value);
        if (!script || !Array.isArray(script.windows) || !script.windows.length) throw new Error(tr('JSON 必须包含至少一个窗口。'));
      }
      showMessage('#infiniteCreateMessage', tr('正在创建项目…'), true);
      const response = await request('/api/v1/infinite-projects', {method:'POST', headers:headers(true), body:JSON.stringify(body)});
      project = await response.json();
      promptDraftProjectId = null;
      if (script) {
        q('#infiniteJsonInput').value = JSON.stringify(script, null, 2);
        showMessage('#infiniteCreateMessage', tr('项目已创建，正在启动自动正式生成队列…'), true);
        const batch = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/batch`, {method:'POST', headers:headers(true), body:JSON.stringify(script)});
        project = await batch.json();
      }
      await loadProjects();
      renderProject();
    } catch (error) {
      if (project) {
        await loadProjects();
        renderProject();
        showMessage('#infiniteJsonMessage', error instanceof SyntaxError ? `${tr('JSON 格式错误：')}${error.message}` : error.message);
      } else {
        showMessage('#infiniteCreateMessage', error instanceof SyntaxError ? `${tr('JSON 格式错误：')}${error.message}` : error.message);
      }
    }
  }

  function appendPayload(form) {
    const data = new FormData();
    for (const name of ['window_description', 'duration_seconds', 'overlap_seconds', 'acceleration', 'visual_memory_resolution', 'seed', 'save_shared_as_default']) {
      data.append(name, form.elements[name].value || (name === 'seed' ? 'random' : ''));
    }
    const memoryEnabled = q('#infiniteMemoryEnabled').checked;
    data.append('visual_memory_capacity', memoryEnabled ? form.elements.visual_memory_capacity.value : '0');
    data.append('audio_memory_capacity', memoryEnabled ? form.elements.audio_memory_capacity.value : '0');
    data.append('inherit_references', project.window_count && form.elements.inherit_references?.checked ? 'true' : 'false');
    const selectedValue = selectedWindowEntry?.kind === 'draft'
      ? promptDrafts[selectedWindowEntry.index]
      : selectedWindowEntry?.kind === 'retry' ? retryDraft : null;
    data.append('excluded_reference_roles', (selectedValue?.excluded_reference_roles || []).join(','));
    data.append('execution_mode', 'complete');
    if (!project.trajectory_locked) {
      data.append('model_variant', project.model_variant || 'base');
      data.append('sampling_steps', String(project.sampling_steps || 20));
      data.append('acceleration', String(project.acceleration || 0));
      data.append('duration_seconds', String(project.window_duration_seconds || 8));
      data.append('overlap_seconds', String(project.overlap_seconds || 1.625));
      data.append('visual_memory_capacity', String(project.visual_memory_capacity || 0));
      data.append('audio_memory_capacity', String(project.audio_memory_capacity || 0));
      data.append('visual_memory_resolution', project.visual_memory_resolution || '360p');
      data.append('resolution', project.preview_resolution || project.resolution || '480p');
      data.append('aspect_ratio', project.aspect_ratio || '16:9');
    }
    if (!q('#infiniteKeyframes').hidden) {
      const firstFrame = form.elements.first_frame?.files?.[0];
      const lastFrame = form.elements.last_frame?.files?.[0];
      if (firstFrame) data.append('first_frame', firstFrame);
      if (lastFrame) data.append('last_frame', lastFrame);
    }
    if (!q('#infiniteReferences').hidden) {
      [...(form.elements.reference_images?.files || [])].slice(0, 9).forEach((file, index) => data.append(`reference_image_${index + 1}`, file));
      [...(form.elements.reference_audios?.files || [])].slice(0, 3).forEach((file, index) => data.append(`reference_audio_${index + 1}`, file));
    }
    return data;
  }

  async function appendWindow(event) {
    event.preventDefault();
    if (!project || !selectedWindowEntry) return;
    saveSelectedPrompt();
    const entry = {...selectedWindowEntry};
    if (entry.kind === 'draft' && entry.index !== 0) return showMessage('#infiniteWindowMessage', tr('请先提交前面的窗口。'));
    try {
      q('#appendInfiniteWindow').disabled = true;
      showMessage('#infiniteWindowMessage', tr('正在提交预览窗口…'), true);
      if (project.can_retry_tail) {
        const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/windows/last`, {method:'DELETE'});
        project = await response.json();
        showMessage('#infiniteWindowMessage', tr('失败尾段已移除，正在重新提交…'), true);
      }
      if (!project.window_count && activeEngine === 'reference' && !event.currentTarget.elements.reference_images.files.length && !event.currentTarget.elements.reference_audios.files.length) throw new Error(tr('Ref2VA 的首窗口至少需要一张参考图片或一段参考音频。'));
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/windows`, {method:'POST', body:appendPayload(event.currentTarget)});
      const result = await response.json();
      project = result.project;
      if (entry.kind === 'draft') promptDrafts.shift(); else retryDraft = null;
      selectedWindowEntry = null;
      persistPromptDrafts();
      event.currentTarget.elements.window_description.value = '';
      showMessage('#infiniteWindowMessage', tr('预览窗口已加入队列。'), true);
      await loadProjects();
    } catch (error) {
      showMessage('#infiniteWindowMessage', error.message);
      renderWindowEditor();
    }
  }

  async function startBatch() {
    if (!project) return;
    try {
      const document = JSON.parse(q('#infiniteJsonInput').value);
      showMessage('#infiniteJsonMessage', tr('正在启动后台自动队列…'), true);
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/batch`, {method:'POST', headers:headers(true), body:JSON.stringify(document)});
      project = await response.json();
      selectedWindowEntry = null;
      renderProject();
      await loadProjects();
    } catch (error) {
      showMessage('#infiniteJsonMessage', error instanceof SyntaxError ? `${tr('JSON 格式错误：')}${error.message}` : error.message);
    }
  }

  async function stopBatch() {
    if (!project) return;
    try {
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/batch`, {method:'DELETE'});
      project = await response.json();
      renderProject();
    } catch (error) {
      showMessage('#infiniteJsonMessage', error.message);
    }
  }

  async function submitFinal(event) {
    event.preventDefault();
    if (!project) return;
    const body = project.workflow_version >= 3
      ? {
          acceleration:Number(event.currentTarget.elements.acceleration.value),
          sigma_scale:Number(event.currentTarget.elements.sigma_scale.value),
        }
      : Object.fromEntries(new FormData(event.currentTarget).entries());
    if (project.workflow_version < 3) {
      delete body.sigma_scale;
      body.method = 'h3';
      body.steps = Number(body.steps);
      body.acceleration = Number(body.acceleration);
    }
    try {
      showMessage('#infiniteFinalMessage', tr('正在提交整片二次采样…'), true);
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/final-sampling`, {method:'POST', headers:headers(true), body:JSON.stringify(body)});
      const job = await response.json();
      project.final_sampling = {job_id:job.id, status:job.status, settings:body, video_url:null};
      renderFinalSampling();
      showMessage('#infiniteFinalMessage', tr('整片二次采样已加入任务队列。'), true);
    } catch (error) {
      showMessage('#infiniteFinalMessage', error.message);
    }
  }

  async function deleteSubmittedTail() {
    if (!project) return;
    try {
      saveSelectedPrompt();
      const response = await request(`/api/v1/infinite-projects/${encodeURIComponent(project.id)}/windows/last`, {method:'DELETE'});
      project = await response.json();
      retryDraft = null;
      selectedWindowEntry = null;
      persistPromptDrafts();
      renderProject();
      showMessage('#infiniteWindowMessage', tr('尾部窗口及对应视频已删除。'), true);
    } catch (error) {
      showMessage('#infiniteWindowMessage', error.message);
    }
  }

  async function deleteProject() {
    if (!project) return;
    const target = project;
    const warning = isEnglish() ? `Delete project “${target.title}”? Generated jobs remain in Tasks.` : `删除项目“${target.title}”？已生成任务仍会保留在任务中心。`;
    if (!window.confirm(warning)) return;
    try {
      clearTimeout(pollTimer);
      await request(`/api/v1/infinite-projects/${encodeURIComponent(target.id)}`, {method:'DELETE'});
      localStorage.removeItem(draftStorageKey(target.id));
      project = null;
      promptDraftProjectId = null;
      await loadProjects();
      if (projects.length) await selectProject(projects[0].id); else newProject();
    } catch (error) {
      showMessage('#infiniteWindowMessage', error.message);
    }
  }

  function newProject() {
    saveSelectedPrompt();
    project = null;
    promptDraftProjectId = null;
    promptDrafts = [];
    selectedWindowEntry = null;
    retryDraft = null;
    finalDefaultsProjectId = null;
    q('#infiniteProjectForm').hidden = false;
    q('#infiniteWorkspace').hidden = true;
    syncProjectModelVariantToggle();
    setProjectCreationMode('online');
    q('#infiniteProjectJsonInput').value = jsonTemplate();
    renderProjectList();
    showMessage('#infiniteCreateMessage', '');
  }

  async function activate(engine, runtimeOptions=null) {
    activeEngine = engine;
    activeOptions = runtimeOptions;
    configureProjectResolution();
    q('#infiniteReferences').hidden = engine !== 'reference';
    q('#infiniteKeyframes').hidden = engine === 'reference';
    try {
      await loadProjects(!project);
      if (!project) newProject();
    } catch (error) {
      showMessage('#infiniteCreateMessage', error.message);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    q('#infiniteProjectForm').addEventListener('submit', createProject);
    q('#infiniteWindowForm').addEventListener('submit', appendWindow);
    q('#infiniteFinalSamplingForm').addEventListener('submit', submitFinal);
    q('#newInfiniteProject').addEventListener('click', newProject);
    q('#deleteInfiniteProject').addEventListener('click', deleteProject);
    q('#addInfiniteWindowPrompt').addEventListener('click', addWindowDraft);
    q('#collapseInfiniteWindowEditor').addEventListener('click', () => { saveSelectedPrompt(); selectedWindowEntry = null; renderWindowEditor(); renderWindowList(); });
    qa('[data-project-mode]').forEach(button => button.addEventListener('click', () => setProjectCreationMode(button.dataset.projectMode)));
    q('#loadInfiniteProjectJsonTemplate').addEventListener('click', () => { q('#infiniteProjectJsonInput').value = jsonTemplate(); });
    q('#loadInfiniteJsonTemplate').addEventListener('click', () => { q('#infiniteJsonInput').value = jsonTemplate(); });
    q('#startInfiniteBatch').addEventListener('click', startBatch);
    q('#stopInfiniteBatch').addEventListener('click', stopBatch);
    q('#infiniteProjectResolutionSlider').addEventListener('input', event => setProjectResolution(event.target.value));
    q('#infiniteProjectForm [name="aspect_ratio"]').addEventListener('change', () => setProjectResolution(q('#infiniteProjectResolutionSlider').value));
    q('#infiniteProjectFinalResolutionSlider').addEventListener('input', event => setProjectFinalResolution(event.target.value));
    qa('[data-infinite-final-resolution]').forEach(button => button.addEventListener('click', () => setProjectFinalResolution(button.dataset.infiniteFinalResolution, true)));
    q('#infiniteProjectForm [name="model_variant"]').addEventListener('change', () => {
      syncProjectModelVariantToggle();
      updateProjectSamplingTrack(true);
    });
    q('#infiniteProjectLoraEnabled').addEventListener('change', event => {
      q('#infiniteProjectForm').elements.model_variant.value = event.target.checked ? 'lora' : 'base';
      syncProjectModelVariantToggle();
      updateProjectSamplingTrack(true);
    });
    q('#infiniteProjectForm [name="sampling_steps"]').addEventListener('input', () => updateProjectSamplingTrack(false));
    q('#infiniteProjectFirstPassSteps').addEventListener('input', () => updateProjectSamplingTrack(false));
    bindFrameDropzone('#infiniteFirstFrame');
    bindFrameDropzone('#infiniteLastFrame');
    q('#infiniteReferenceFiles').addEventListener('change', event => distributeInfiniteReferenceFiles(event.target.files));
    if (typeof window.bindFileDrop === 'function') {
      window.bindFileDrop(q('#infiniteReferenceFileDrop'), q('#infiniteReferenceFiles'), {
        maxFiles:12,
        accept:file => file.type.startsWith('image/') || file.type.startsWith('audio/') || /\.(wav|mp3|flac|m4a|ogg|opus)$/i.test(file.name),
      });
    }
    q('#infiniteWindowForm').elements.inherit_references.addEventListener('change', renderWindowEditor);
    for (const control of q('#infiniteWindowForm').querySelectorAll('textarea,input:not([type="file"]),select')) {
      control.addEventListener('input', () => { updateWindowOutputs(); saveSelectedPrompt(); });
      control.addEventListener('change', () => { updateWindowOutputs(); saveSelectedPrompt(); });
    }
    bindRange('#infiniteWindowForm [name="acceleration"]', '#infiniteWindowAccelerationValue', String);
    bindRange('#infiniteFinalSamplingForm [name="acceleration"]', '#infiniteFinalAccelerationValue', String);
    bindRange('#infiniteFinalSamplingForm [name="sigma_scale"]', '#infiniteFinalSigmaScaleValue', value => Number(value).toFixed(2));
    q('#infiniteJsonInput').value = jsonTemplate();
    q('#infiniteProjectJsonInput').value = jsonTemplate();
    syncProjectModelVariantToggle();
    updateProjectSamplingTrack(false);
    setProjectCreationMode('online');
    switchInputMode();
  });

  window.addEventListener('h3serve:locale-changed', () => {
    syncProjectModelVariantToggle();
    renderProjectList();
    if (project) renderProject();
  });

  window.H3InfiniteStudio = {activate, refresh:refreshSelected};
})();
