/* Keeps one reference image in localStorage so every tool can pick it up.
   All file:// pages share the same origin, so this works without a server. */
(function(global){

  const KEY = 'painterTools.sharedImage';
  const MAX_DIM = 1600;
  const QUALITIES = [0.92, 0.8, 0.65, 0.5];

  let bar = null;
  let thumbEl = null;
  let textEl = null;
  let newBtnEl = null;
  let onNewImage = null;

  function read(){
    try{
      return JSON.parse(localStorage.getItem(KEY) || 'null');
    }catch(err){
      return null;
    }
  }

  function scaled(img){
    let w = img.naturalWidth || img.width;
    let h = img.naturalHeight || img.height;

    const scale = Math.min(1, MAX_DIM / Math.max(w, h));
    w = Math.round(w * scale);
    h = Math.round(h * scale);

    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    c.getContext('2d').drawImage(img, 0, 0, w, h);
    return c;
  }

  function set(img, name){
    const c = scaled(img);

    // step the quality down until the data URL fits in the storage quota
    for(const q of QUALITIES){
      const dataUrl = c.toDataURL('image/jpeg', q);
      try{
        localStorage.setItem(KEY, JSON.stringify({
          dataUrl,
          name: name || 'image',
          at: Date.now()
        }));
        showBar(dataUrl, name);
        return true;
      }catch(err){
        if(err.name !== 'QuotaExceededError' && err.code !== 22) return false;
      }
    }
    return false;
  }

  function clear(){
    try{ localStorage.removeItem(KEY); }catch(err){}
    if(bar) bar.hidden = true;
  }

  function buildBar(){
    const wrap = document.querySelector('.wrap');
    if(!wrap || bar) return;

    bar = document.createElement('div');
    bar.className = 'shared-bar';
    bar.hidden = true;

    thumbEl = document.createElement('img');
    thumbEl.className = 'shared-bar__thumb';
    thumbEl.alt = '';

    textEl = document.createElement('span');
    textEl.className = 'shared-bar__text';

    const newBtn = document.createElement('button');
    newBtn.type = 'button';
    newBtn.className = 'btn';
    newBtn.textContent = 'Load different image';
    newBtn.hidden = !onNewImage;
    newBtn.addEventListener('click', () => {
      if(onNewImage) onNewImage();
    });
    newBtnEl = newBtn;

    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.className = 'btn';
    clearBtn.textContent = 'Forget image';
    clearBtn.addEventListener('click', () => {
      clear();
      if(onNewImage) onNewImage();
    });

    bar.append(thumbEl, textEl, newBtn, clearBtn);

    const nav = wrap.querySelector('.tool-nav');
    wrap.insertBefore(bar, nav ? nav.nextSibling : wrap.firstChild);
  }

  function showBar(dataUrl, name){
    buildBar();
    if(!bar) return;

    thumbEl.src = dataUrl;
    textEl.textContent = name
      ? `Shared image: ${name} — available in every tool`
      : 'Shared image — available in every tool';
    bar.hidden = false;
  }

  /* onImage: called with a loaded <img> when a shared image already exists.
     onNewImage: called when the user wants to pick a different image. */
  function attach(opts){
    opts = opts || {};
    onNewImage = opts.onNewImage || null;

    buildBar();
    if(newBtnEl) newBtnEl.hidden = !onNewImage;

    const saved = read();
    if(!saved || !saved.dataUrl) return;

    showBar(saved.dataUrl, saved.name);

    if(opts.onImage){
      const img = new Image();
      img.onload = () => opts.onImage(img, saved.name);
      img.src = saved.dataUrl;
    }
  }

  global.SharedImage = { set, clear, attach, read };

})(window);
