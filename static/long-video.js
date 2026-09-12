'use strict';
const el = id => document.getElementById(id);
let busy = false;
let schemaVersion = 2;
const resolutionStops = ['360p','480p','540p','720p','900p','1080p'];
function status(message, error=false) { el('status').textContent=message; el('status').className=error?'error':''; }
function headers() { return el('apiKey').value ? {'X-API-Key':el('apiKey').value} : {}; }
async function request(url, init={}) {
  const response=await fetch(url,{...init,headers:{...headers(),...(init.headers||{})}});
  if(!response.ok) throw new Error(await response.text());
  return response.json();
}
async function example(name) {
  const value=await request(`/static/long-video-examples/${name}.json`);
  schemaVersion=value.long_video.version;
  el('story').value=JSON.stringify(value.long_video.story,null,2);
  el('overlap').value=value.long_video.overlap_seconds;el('maximum').value=value.long_video.max_window_seconds;
  el('memory').value=value.long_video.memory;el('memoryValue').value=el('memory').value;el('seed').value=value.seed;
  el('preview').replaceChildren();status('模板已载入。可修改故事并预览实际窗口。');
}
function payload() {
  if(!el('editor').reportValidity()) throw new Error('请补齐有效输入。');
  const form=new FormData();
  form.set('long_video',JSON.stringify({version:schemaVersion,overlap_seconds:Number(el('overlap').value),max_window_seconds:Number(el('maximum').value),memory:Number(el('memory').value),story:JSON.parse(el('story').value)}));
  for(const [key,id] of Object.entries({model_variant:'variant',resolution:'resolution',aspect_ratio:'ratio',sampling_steps:'steps',acceleration:'acceleration',seed:'seed'})) form.set(key,el(id).value);
  if(el('first').files[0]) form.set('first_frame',el('first').files[0]);
  if(el('last').files[0]) form.set('last_frame',el('last').files[0]);
  if(el('pictures').files.length>9 || el('audios').files.length>3) throw new Error('参考素材超过数量上限。');
  Array.from(el('pictures').files).forEach((file,i)=>form.set(`reference_image_${i+1}`,file));
  Array.from(el('audios').files).forEach((file,i)=>form.set(`reference_audio_${i+1}`,file));
  return form;
}
function showPreview(value) {
  const container=el('preview');container.replaceChildren();
  const summary=document.createElement('section'),budget=value.memory_budget;
  const h=document.createElement('h2');h.textContent='实际执行计划';summary.append(h);
  const p=document.createElement('p');p.textContent=`${value.windows.length} 个窗口，总长 ${value.actual_duration_seconds.toFixed(3)} 秒；重叠 ${value.effective_overlap_seconds.toFixed(3)} 秒；单窗上限 ${value.effective_max_window_seconds.toFixed(3)} 秒。长期视觉参考最多 ${budget.video_frames} 帧，自动声音参考 ${budget.audio_clips} 段 × ${budget.audio_seconds.toFixed(3)} 秒，最多增加 ${budget.maximum_memory_tokens} 个媒体 token。`;summary.append(p);
  if(budget.automatic_audio_reason==='single_speaker_only') {const note=document.createElement('p');note.textContent='自动声音记忆只用于明确的单说话人任务；多人任务请使用对应的用户声音参考。';summary.append(note);}
  container.append(summary);
  for(const [i,window] of value.windows.entries()) {
    const section=document.createElement('section'),title=document.createElement('h3'),pre=document.createElement('pre');
    title.textContent=`窗口 ${i+1} · ${window.shot_id}/${window.segment_id} · ${{opening:'开场',cut:'切换镜头',continue:'延续镜头'}[window.transition]} · 新内容 ${window.actual_start_seconds.toFixed(3)}–${window.actual_end_seconds.toFixed(3)} 秒`;
    pre.textContent=window.prompt;section.append(title,pre);container.append(section);
  }
}
async function run(generate) {
  if(busy) return;busy=true;el('previewButton').disabled=el('generateButton').disabled=true;
  try {
    const body=payload();status('正在检查引用、窗口容量和时间安排……');
    const preview=await request('/api/v1/long-video/preview',{method:'POST',body});showPreview(preview);
    if(!generate) {status('校验通过。上方显示每个窗口实际接收的提示词。');return;}
    // Reuse the exact validated submission rather than re-reading editable fields.
    const job=await request('/api/v1/generations',{method:'POST',body});
    const jobURL=`/api/v1/jobs/${encodeURIComponent(job.id)}`;
    el('jobLink').href=jobURL;status(`任务已提交：${job.id}`);
    for(;;) {
      const state=await request(jobURL);
      status(`${state.status} · ${state.progress?.detail||''} ${state.progress?.percent?.toFixed(1)||0}%`);
      if(state.status==='succeeded') {el('result').hidden=false;el('video').src=state.video_url;el('download').href=state.video_url;status('视频已生成，请连续观看并审核声音。');break;}
      if(['failed','cancelled','canceled'].includes(state.status)) throw new Error(state.error||state.status);
      await new Promise(resolve=>setTimeout(resolve,3000));
    }
  } catch(error) {status(error.message,true);}
  finally {busy=false;el('previewButton').disabled=el('generateButton').disabled=false;}
}
el('memory').addEventListener('input',()=>{el('memoryValue').value=el('memory').value;});
function setResolution(value) {
  if(!resolutionStops.includes(value)) return;
  el('resolution').value=value;el('resolutionSlider').value=String(resolutionStops.indexOf(value));
  el('resolutionValue').value=value.toUpperCase();
  document.querySelectorAll('[data-resolution]').forEach(button=>button.classList.toggle('active',button.dataset.resolution===value));
}
el('resolutionSlider').addEventListener('input',event=>setResolution(resolutionStops[Number(event.target.value)]));
document.querySelectorAll('[data-resolution]').forEach(button=>button.addEventListener('click',()=>setResolution(button.dataset.resolution)));
setResolution('480p');
el('previewButton').addEventListener('click',()=>run(false));
el('editor').addEventListener('submit',event=>{event.preventDefault();run(true);});
document.querySelectorAll('[data-example]').forEach(button=>button.addEventListener('click',()=>example(button.dataset.example).catch(error=>status(error.message,true))));
example('cafe30').catch(error=>status(error.message,true));
