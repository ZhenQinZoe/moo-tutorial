import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import MarkdownIt from 'markdown-it';
import footnote from 'markdown-it-footnote';
import mark from 'markdown-it-mark';
import sub from 'markdown-it-sub';
import sup from 'markdown-it-sup';
import container from 'markdown-it-container';

const root = path.dirname(new URL(import.meta.url).pathname);
const out = path.join(root, 'dist');
const notes = JSON.parse(await fs.readFile(path.join(root, 'content/manifest.json'), 'utf8'));
const assets = JSON.parse(await fs.readFile(path.join(root, 'assets/manifest.json'), 'utf8').catch(() => '{}'));
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const md = new MarkdownIt({html: true, linkify: true, typographer: false}).use(footnote).use(mark).use(sub).use(sup);

// Consume TeX before Markdown processes its underscores, backslashes, and pipes.
function mathToken(state, silent) {
  const p = state.pos;
  const opens = ['$$', '\\[', '\\(', '$'];
  const open = opens.find(o => state.src.startsWith(o, p));
  if (!open || (open === '$' && /\s/.test(state.src[p + 1] || ' '))) return false;
  const close = ({'\\[':'\\]', '\\(':'\\)'})[open] || open;
  let end = state.src.indexOf(close, p + open.length);
  while (end !== -1 && state.src[end - 1] === '\\') end = state.src.indexOf(close, end + close.length);
  if (end < 0 || (open === '$' && /\s/.test(state.src[end - 1]))) return false;
  if (!silent) {
    const t = state.push('math_inline', '', 0);
    t.content = state.src.slice(p + open.length, end);
    t.meta = {display: open === '$$' || open === '\\['};
  }
  state.pos = end + close.length;
  return true;
}
md.inline.ruler.before('escape', 'math_inline', mathToken);
md.inline.ruler.before('sup', 'block_anchor', (state,silent) => {
  const match=state.src.slice(state.pos).match(/^\^([A-Za-z][\w:-]*)(?=\s*(?:\n|$))/);
  if(!match)return false;
  if(!silent){const t=state.push('block_anchor','',0);t.content=match[1];}
  state.pos+=match[0].length;return true;
});
md.renderer.rules.block_anchor=(tokens,i)=>`<span class="block-anchor" id="^${esc(tokens[i].content)}"></span>`;
md.renderer.rules.math_inline = (tokens, i) => `<span class="tex${tokens[i].meta.display ? ' display-tex' : ''}">${tokens[i].meta.display ? '\\[' : '\\('}${esc(tokens[i].content)}${tokens[i].meta.display ? '\\]' : '\\)'}</span>`;
md.block.ruler.before('fence', 'math_block', (state, start, end, silent) => {
  const pos = state.bMarks[start] + state.tShift[start];
  const open = state.src.startsWith('$$', pos) ? '$$' : state.src.startsWith('\\[', pos) ? '\\[' : null;
  if (!open) return false;
  const close = open === '$$' ? '$$' : '\\]';
  const finish = state.src.indexOf(close, pos + 2);
  if (finish < 0) return false;
  let last = start;
  while (last < end && state.eMarks[last] < finish + 2) last++;
  if (last >= end || state.src.slice(finish + 2, state.eMarks[last]).trim()) return false;
  if (silent) return true;
  const t = state.push('math_block', '', 0);
  t.block = true; t.content = state.src.slice(pos + 2, finish); t.map = [start, last + 1];
  state.line = last + 1;
  return true;
}, {alt: ['paragraph', 'blockquote', 'list']});
md.renderer.rules.math_block = (tokens, i) => `<div class="math-block">\\[${esc(tokens[i].content)}\\]</div>\n`;
md.use(container, 'callout', {
  validate: params => /^(?:info|warning|success|danger|note|tip|\{)/.test(params.trim()),
  render: (tokens, i) => {
    if (tokens[i].nesting < 0) return '</div>\n';
    const p = tokens[i].info.trim();
    const cls = p.startsWith('{') ? [...p.matchAll(/\.([\w-]+)/g)].map(m => m[1]).join(' ') : p.split(/\s/)[0];
    const id = p.match(/#([\w-]+)/)?.[1];
    const style = p.match(/style="([^"]+)"/)?.[1];
    return `<div class="callout ${esc(cls)}"${id ? ` id="${esc(id)}"` : ''}${style ? ` style="${esc(style)}"` : ''}>\n`;
  }
});
const fence = md.renderer.rules.fence;
md.renderer.rules.fence = (t,i,o,e,s) => t[i].info.trim() === 'mermaid' ? `<pre class="mermaid">${esc(t[i].content)}</pre>\n` : fence(t,i,o,e,s);
const imageRule = md.renderer.rules.image;
md.renderer.rules.image = (t,i,o,e,s) => {
  const original = t[i].attrGet('src');
  t[i].attrSet('src', assets[original]?.file ? `assets/${assets[original].file}` : assets[original]?.url || original);
  t[i].attrSet('loading', 'eager'); t[i].attrSet('decoding', 'async');
  return imageRule(t,i,o,e,s);
};
const textOf = token => (token.children || []).map(t => t.type === 'image' ? t.content : t.content || '').join('');
const slugify = text => text.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g,'-');
const linkFor = n => n.slug === 'overview' ? 'index.html' : `${n.slug}.html`;
const report = {sourceFiles: [], missingImages: [], pages: []};

for (const n of notes) {
  const raw = await fs.readFile(path.join(root, 'content', `${n.id}.md`));
  const hash = crypto.createHash('sha256').update(raw).digest('hex');
  if (hash !== n.sha256) throw new Error(`Source changed: ${n.title}`);
  n.raw = raw.toString('utf8');
  n.body = n.raw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '');
  n.tokens = md.parse(n.body, {});
  const used = new Map(); n.headings = [];
  n.tokens.forEach((t,i) => {
    if (t.type !== 'heading_open') return;
    const next = n.tokens[i+1]; const text = textOf(next);
    let id = slugify(text) || 'section'; const count = used.get(id) || 0; used.set(id,count+1);
    if (count) id += `-${count+1}`;
    t.attrSet('id',id); n.headings.push({level:Number(t.tag.slice(1)),text,id,html:md.renderInline(next.content)});
  });
  n.heading = n.headings.find(h=>h.level===1)?.text || n.title;
  n.toc = n.headings.filter(h=>h.level===2 || h.level===3);
  report.sourceFiles.push({file:`content/${n.id}.md`,sha256:hash,verified:true});
  for (const m of n.raw.matchAll(/!\[[^\]]*\]\(([^\s)]+)[^)]*\)/g)) {
    if (!assets[m[1]]?.file) report.missingImages.push({page:n.slug,source:m[1]});
  }
}
const rawLink = md.renderer.rules.link_open || ((t,i,o,e,s)=>s.renderToken(t,i,o));
const noteAliases={
  'Intro Deterministic MGDA.md':'ROaYhbMiQ3OzhgIWLNnL-w',
  'Intro MOO -- Deterministic MGDA.md':'ROaYhbMiQ3OzhgIWLNnL-w',
  'Intro Stochastic MGDA.md':'G5i2O6U3TR67PEcyndCEcg'
};
md.renderer.rules.link_open = (t,i,o,e,s) => {
  const href = t[i].attrGet('href') || '';
  const name=decodeURIComponent(href.split(/[?#]/)[0]);
  const n = notes.find(n=>name===n.url || name===n.title || n.id===noteAliases[name]);
  if(n) t[i].attrSet('href',linkFor(n)+(href.includes('#') ? '#'+href.split('#')[1] : ''));
  return rawLink(t,i,o,e,s);
};
function sectionNav(n,current) {
  const active=n.slug===current.slug;
  const children=n.headings.filter(h=>h.level===2 && /^\d/.test(h.text));
  return `<li class="chapter ${active?'current':''}"><div class="chapter-row"><a href="${linkFor(n)}"${active?' aria-current="page"':''}>${esc(n.heading)}</a>${children.length ? `<button class="expand" aria-label="${esc('Sections in '+n.heading)}" aria-expanded="${active}"><span></span></button>` : ''}</div>${children.length?`<ul class="subnav"${active?'':' hidden'}>${children.map(h=>`<li><a href="${linkFor(n)}#${h.id}">${h.html}</a></li>`).join('')}</ul>`:''}</li>`;
}
function navigation(current) {
  return `<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32"><path d="M5 5v22h22M8 22c4-1 6-4 8-8s4-6 9-6"/><circle cx="9" cy="22" r="2"/><circle cx="16" cy="14" r="2"/><circle cx="25" cy="8" r="2"/></svg></span><span>MOO Tutorial</span></a><div class="search-wrap"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></svg><input id="search" type="search" placeholder="Search tutorial" aria-label="Search tutorial" autocomplete="off"><kbd>/</kbd></div><div id="search-results" aria-live="polite" hidden></div><nav aria-label="Chapters" id="chapter-nav"><div class="nav-label">CONTENTS</div><ul class="chapter-list">${notes.filter(n=>n.group==='chapters').map(n=>sectionNav(n,current)).join('')}</ul><div class="nav-secondary"><a class="${current.slug==='outline'?'selected':''}" href="outline.html">Outline</a><details${current.group==='notes'?' open':''}><summary>Foundation notes</summary><ul>${notes.filter(n=>n.group==='notes').map(n=>`<li><a href="${linkFor(n)}"${n.slug===current.slug?' aria-current="page"':''}>${esc(n.heading)}</a></li>`).join('')}</ul></details></div></nav><div class="sidebar-bottom"><a href="https://hackmd.io/team/moosurvey?nav=overview" target="_blank" rel="noopener">HackMD source</a><span>Multi-objective optimization</span></div>`;
}
function render(n) {
  const toc = `<nav class="inline-toc" aria-label="Section contents"><ul>${n.toc.filter(h=>h.level===2).map(h=>`<li><a href="#${h.id}">${h.html}</a></li>`).join('')}</ul></nav>`;
  let html = md.renderer.render(n.tokens,md.options,{}).replace(/<p>\[TOC\]<\/p>/g,toc);
  const main = notes.filter(n=>n.group==='chapters'); const idx=main.indexOf(n);
  const pages = idx<0 ? '' : `<nav class="page-navigation" aria-label="Adjacent chapters">${idx>0?`<a href="${linkFor(main[idx-1])}"><span>Previous chapter</span><strong>${esc(main[idx-1].heading)}</strong></a>`:'<span></span>'}${idx<main.length-1?`<a href="${linkFor(main[idx+1])}" class="next"><span>Next chapter</span><strong>${esc(main[idx+1].heading)}</strong></a>`:''}</nav>`;
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>${esc(n.heading)} | MOO Tutorial</title><link rel="icon" href="favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="theme.css"><script src="math-config.js"></script><script defer src="vendor/mathjax/tex-chtml-full.js"></script><script defer src="vendor/mermaid.min.js"></script><script defer src="search-index.js"></script><script defer src="app.js"></script></head>
<body data-page="${esc(n.slug)}"><a class="skip-link" href="#main">Skip to content</a><header class="mobile-header"><button id="menu-toggle" aria-label="Open navigation" aria-controls="sidebar" aria-expanded="false"><span></span><span></span><span></span></button><a href="index.html">MOO Tutorial</a></header><div class="mobile-backdrop" hidden></div><aside id="sidebar" class="sidebar">${navigation(n)}</aside><div class="reading-progress" aria-hidden="true"></div><main id="main" class="main"><div class="page-top"><div class="breadcrumb"><a href="index.html">MOO Tutorial</a><span>/</span><span>${n.group==='notes'?'Foundation notes':n.slug==='overview'?'Overview':n.slug==='outline'?'Outline':`Chapter ${idx}`}</span></div><a class="source-link" href="${n.url}" target="_blank" rel="noopener">View source</a></div><article class="article">${html}</article>${pages}<footer class="page-footer"><span>MOO Tutorial</span><a href="#main">Back to top</a></footer></main><aside class="page-toc" aria-label="On this page"><div class="nav-label">ON THIS PAGE</div><nav>${n.toc.map(h=>`<a class="toc-level-${h.level}" href="#${h.id}">${h.html}</a>`).join('')}</nav></aside><div id="image-viewer" role="dialog" aria-modal="true" aria-label="Figure viewer" hidden><button aria-label="Close figure">×</button><img alt=""></div></body></html>`;
}
await fs.mkdir(out,{recursive:true});
for (const file of ['theme.css','app.js','math-config.js','favicon.svg']) await fs.copyFile(path.join(root,file),path.join(out,file));
await fs.cp(path.join(root,'assets'),path.join(out,'assets'),{recursive:true});
await fs.mkdir(path.join(out,'vendor'),{recursive:true});
await fs.cp(path.join(root,'node_modules/mathjax/es5'),path.join(out,'vendor/mathjax'),{recursive:true});
await fs.copyFile(path.join(root,'node_modules/mermaid/dist/mermaid.min.js'),path.join(out,'vendor/mermaid.min.js'));
let licenses='Third-party notices for the distributed website\n\n';
for(const name of ['mathjax','mermaid']){
  const pkg=JSON.parse(await fs.readFile(path.join(root,'node_modules',name,'package.json'),'utf8'));
  licenses+=`\n${name} ${pkg.version} — ${pkg.license}\n`;
  for(const file of ['LICENSE','LICENSE.txt','LICENSE.md','LICENSES.txt']){
    const text=await fs.readFile(path.join(root,'node_modules',name,file),'utf8').catch(()=>null);
    if(text){licenses+=text+'\n';break;}
  }
}
await fs.writeFile(path.join(out,'THIRD-PARTY-NOTICES.txt'),licenses);
const search=[];
for (const n of notes) {
  await fs.writeFile(path.join(out,linkFor(n)),render(n));
  report.pages.push({file:linkFor(n),title:n.heading,headings:n.headings.length});
  // Every result excerpt is copied from the original source.
  const sections=n.body.split(/(?=^#{1,3} )/m);
  for (const section of sections) {
    if(!section.trim()) continue;
    const heading=section.match(/^#{1,3}\s+(.+)/)?.[1];
    const matched=n.headings.find(h=>h.text===textOf(md.parseInline(heading||'',{})[0]));
    search.push({title:matched?.text||n.heading,chapter:n.heading,url:linkFor(n)+(matched?'#'+matched.id:''),text:section.replace(/<[^>]*>/g,' ').replace(/!\[[^\]]*\]\([^)]*\)/g,' ').replace(/[#*`\[\]]/g,'').replace(/\s+/g,' ').trim()});
  }
}
await fs.writeFile(path.join(out,'search-index.js'),'window.MOO_SEARCH='+JSON.stringify(search).replace(/</g,'\\u003c')+';');
await fs.writeFile(path.join(out,'.nojekyll'),'');
await fs.writeFile(path.join(root,'validation.json'),JSON.stringify(report,null,2)+'\n');
console.log(`Built ${notes.length} pages. Verified all ${notes.length} source checksums. ${report.missingImages.length} unresolved image references.`);
