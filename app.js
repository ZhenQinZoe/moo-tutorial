(() => {
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const mobileHeader=$('.mobile-header');
  const syncHeaderHeight=()=>{
    const height=mobileHeader.getBoundingClientRect().height;
    if(height)document.documentElement.style.setProperty('--mobile-header-height',`${Math.ceil(height)}px`);
    else document.documentElement.style.removeProperty('--mobile-header-height');
  };
  new ResizeObserver(syncHeaderHeight).observe(mobileHeader);
  syncHeaderHeight();
  const followOldReference = () => {
    const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if(target?.dataset.referenceTarget)location.replace(target.dataset.referenceTarget);
  };
  followOldReference(); window.addEventListener('hashchange',followOldReference);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  $$('.expand').forEach(button=>button.addEventListener('click',()=>{
    const open=button.getAttribute('aria-expanded')!=='true';
    button.setAttribute('aria-expanded',String(open));
    button.closest('.chapter').querySelector('.subnav').hidden=!open;
  }));
  const setMenu = open => {
    $('#sidebar').classList.toggle('open',open); $('.mobile-backdrop').classList.toggle('open',open);
    $('.mobile-backdrop').hidden=!open; $('#menu-toggle').setAttribute('aria-expanded',String(open));
    $('#menu-toggle').setAttribute('aria-label',open?'Close navigation':'Open navigation');
  };
  $('#menu-toggle').addEventListener('click',()=>setMenu(!$('#sidebar').classList.contains('open')));
  $('.mobile-backdrop').addEventListener('click',()=>setMenu(false));
  $$('#sidebar a').forEach(a=>a.addEventListener('click',()=>setMenu(false)));
  $$('.article table').forEach(t=>{const w=document.createElement('div');w.className='table-wrap';w.tabIndex=0;w.setAttribute('role','region');w.setAttribute('aria-label','Scrollable table');t.before(w);w.append(t)});
  const labelScrollableMath=()=>{
    // Keep inline formulas in the text flow, even on narrow screens.
    const inline=$$('.tex:not(.display-tex)');
    inline.forEach(el=>el.style.removeProperty('font-size'));
    const sizes=inline.map(el=>{
      let parent=el.parentElement;
      while(parent&&(!parent.clientWidth||['inline','contents'].includes(getComputedStyle(parent).display)))parent=parent.parentElement;
      if(!parent)return null;
      const style=getComputedStyle(parent);
      const available=parent.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
      const width=el.getBoundingClientRect().width;
      return available>0&&width>available+1?100*available/width:null;
    });
    inline.forEach((el,i)=>{if(sizes[i])el.style.fontSize=`${sizes[i]}%`});
    $$('.display-tex,.math-block').forEach(el=>{
      const overflow=el.scrollWidth>el.clientWidth+1;
      if(overflow){el.tabIndex=0;el.setAttribute('role','region');el.setAttribute('aria-label','Scrollable equation');}
      else {el.removeAttribute('tabindex');el.removeAttribute('role');el.removeAttribute('aria-label');}
    });
  };
  window.addEventListener('math-ready',labelScrollableMath);
  window.addEventListener('resize',labelScrollableMath);
  if(document.documentElement.dataset.mathReady)labelScrollableMath();
  const viewer=$('#image-viewer'); let lastImage=null;
  function closeImage(){viewer.hidden=true;document.body.classList.remove('image-open');lastImage?.focus();}
  $$('.article img').forEach(img=>{
    const markBroken=()=>{img.classList.add('broken-image');img.title='This image is unavailable at its original source.'};
    img.addEventListener('error',markBroken);if(img.complete&&!img.naturalWidth)markBroken();
    img.tabIndex=0;
    const open=()=>{if(!img.naturalWidth)return;lastImage=img;viewer.querySelector('img').src=img.src;viewer.querySelector('img').alt=img.alt;viewer.hidden=false;document.body.classList.add('image-open');viewer.querySelector('button').focus()};
    img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter')open()});
  });
  viewer.querySelector('button').addEventListener('click',closeImage);
  viewer.addEventListener('click',e=>{if(e.target===viewer)closeImage()});
  viewer.addEventListener('keydown',e=>{if(e.key==='Tab'){e.preventDefault();viewer.querySelector('button').focus()}});
  const search=$('#search'), results=$('#search-results'), chapterNav=$('#chapter-nav');
  let timer;
  search.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>{
    const query=search.value.trim().toLowerCase();results.hidden=!query;chapterNav.hidden=!!query;
    if(!query){results.innerHTML='';return;}
    const terms=query.split(/\s+/).filter(Boolean);
    const matches=(window.MOO_SEARCH||[]).map(r=>{const title=r.title.toLowerCase(),text=r.text.toLowerCase();return {...r,score:terms.every(t=>text.includes(t)||title.includes(t))?terms.reduce((s,t)=>s+(title.includes(t)?10:1),0):0}}).filter(r=>r.score).sort((a,b)=>b.score-a.score).slice(0,30);
    results.innerHTML=`<div class="search-count">${matches.length?`${matches.length} results`:'No results'}</div>`+matches.map(r=>{
      const pos=r.text.toLowerCase().indexOf(terms[0]);const start=Math.max(0,pos-45);const excerpt=(start?'…':'')+r.text.slice(start,start+160)+'…';
      return `<a href="${esc(r.url)}"><small>${esc(r.chapter)}</small><strong>${esc(r.title)}</strong><p>${esc(excerpt)}</p></a>`;
    }).join('');
  },120)});
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'){closeImage();setMenu(false);if(document.activeElement===search){search.value='';search.dispatchEvent(new Event('input'));search.blur()}}
    if(e.key==='/'&&!/INPUT|TEXTAREA/.test(e.target.tagName)){e.preventDefault();if(innerWidth<=760)setMenu(true);search.focus()}
  });
  function progress(){const max=document.documentElement.scrollHeight-innerHeight;$('.reading-progress').style.width=(Math.max(0,scrollY)/Math.max(1,max))*(innerWidth-(innerWidth>1200?300:innerWidth>760?270:0))+'px'}
  window.addEventListener('scroll',progress,{passive:true});window.addEventListener('resize',progress);progress();
  const tocLinks=$$('.page-toc a');const headings=$$('.article h2,.article h3');
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(!visible.length)return;const id=visible[0].target.id;tocLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+id));$$('.current .subnav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+id))},{rootMargin:'-5% 0px -65% 0px'});headings.forEach(h=>observer.observe(h))}
  if(window.mermaid){mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'neutral',fontFamily:'Arial, sans-serif',flowchart:{htmlLabels:true,useMaxWidth:true}});mermaid.run({querySelector:'.mermaid'}).catch(error=>console.error('Diagram rendering:',error.message))}
})();
