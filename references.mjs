// Bibliography and citation overlays. The HackMD Markdown files are immutable.
import fs from 'node:fs/promises';

const escapeRE = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
export function paperIdentity(url) {
  const s = decodeURIComponent(url).replace(/&amp;/g, '&');
  const arxiv = s.match(/(?:arxiv\.org\/(?:abs|pdf|html)\/|10\.48550\/arxiv\.)(\d{4}\.\d{4,5})(?:v\d+)?/i);
  if (arxiv) return `arxiv:${arxiv[1]}`;
  const openreview = s.match(/openreview\.net\/(?:forum|pdf)\?id=([^&#]+)/);
  if (openreview) return `openreview:${openreview[1]}`;
  const proceedings = s.match(/(?:papers|proceedings)\.(?:neurips|nips|iclr)\.cc\/paper(?:_files\/paper)?\/(\d{4})\/(?:hash|file)\/([a-f\d]{32})/);
  if (proceedings) return `proceedings:${s.includes('iclr.cc')?'iclr':'neurips'}:${proceedings[1]}:${proceedings[2]}`;
  const pmlr = s.match(/(?:proceedings|proceedings\.mlr)\.press\/(v\d+)\/([\w-]+)/);
  if (pmlr) return `pmlr:${pmlr[1]}:${pmlr[2]}`;
  if(s.includes('openaccess.thecvf.com'))return s.replace('/papers/','/html/').replace(/\.pdf(?:#.*)?$/,'.html');
  const doi = s.match(/(?:doi\.org\/|link\.springer\.com\/(?:article|chapter|book)\/)(10\.[^?#]+)/i);
  if (doi) return `doi:${doi[1].replace(/\.pdf$/, '').toLowerCase()}`;
  return s.replace(/^http:/, 'https:').replace(/^https:\/\/www\./, 'https://').replace(/[?#].*$/, '').replace(/\/$/, '');
}

function surnames(authors) {
  const initials = [...authors.matchAll(/(?:^|,\s*|&\s*)([\p{L}\p{M}'’\-]+(?:\s+[\p{L}\p{M}'’\-]+)?),\s*(?:[A-Z]\.)/gu)].map(m => m[1]);
  if (initials.length) return initials;
  return authors.replace(/,?\s*et al\.?$/, '').split(/,?\s+(?:and|&)\s+|,\s*/).filter(Boolean).map(a => a.trim().split(/\s+/).at(-1)).filter(a => !/^[A-Z]\.$/.test(a));
}

function authorForms(names) {
  if(!names.length)return [];
  const forms=[names[0],`${names[0]} et al.`,`${names[0]} et al`];
  if(names.length===2)forms.push(names.join(' and '),names.join(' & '));
  if(names.length>2)forms.push(names.slice(0,-1).join(', ')+', and '+names.at(-1),names.slice(0,-1).join(', ')+' and '+names.at(-1));
  return forms;
}

export async function loadReferences(root, esc) {
  const extracted = JSON.parse(await fs.readFile(`${root}/references/extracted.json`, 'utf8'));
  const catalog = JSON.parse(await fs.readFile(`${root}/references/verified.json`, 'utf8'));
  const merges = catalog.merges || {};
  const canonical = key => merges[key] || key;
  const records = catalog.records.filter(r => !merges[r.key]);
  const sorted = [...records].sort((a,b)=>a.authors[0].split(' ').at(-1).localeCompare(b.authors[0].split(' ').at(-1)) || a.year-b.year || a.title.localeCompare(b.title));
  const numbers = new Map(sorted.map((r,i)=>[r.key,i+1]));
  const byKey = new Map(records.map(r => [r.key, r]));
  const names=new Set(extracted.entries.flatMap(e=>authorForms(surnames(e.authors))));
  for(const r of records)for(const name of authorForms(r.authors.map(a=>a.split(' ').at(-1))))names.add(name);
  const authorPattern=[...names].sort((a,b)=>b.length-a.length).map(escapeRE).join('|');
  const authorStartPattern=new RegExp(`^(${authorPattern})(?=\\s*(?:[, (]|\\d|$))`,'u');
  const authorYearPattern=new RegExp(`(?<![\\p{L}])(?:\\(\\s*)?(?:${authorPattern})\\s*(?:,\\s*|\\(\\s*|\\s+)(?:18|19|20)\\d{2}[a-z]?(?:/\\d{4})?\\)?`, 'gu');
  function withoutAuthorYear(label) {
    if(label==='https://hackmd.io/team/moosurveyXu')return '';
    let clean=label.replace(authorYearPattern,'').trim();
    if(names.has(clean))clean='';
    clean=clean.replace(/\(\s*\)/g,'').replace(/,\s*\)/g,')').replace(/^[,;]\s*/,'').trim();
    // A pinpoint locator may follow a removed author/year inside parentheses.
    if((clean.match(/\)/g)||[]).length>(clean.match(/\(/g)||[]).length)clean=clean.replace(/\)\s*$/,'');
    return clean.replace(/\s+of\s*$/,'').trim();
  }
  for(const r of records)if(r.existence!=='confirmed'||!r.url||!r.title||!r.authors?.length||!r.evidence?.length)throw new Error(`Incomplete reference record: ${r.key}`);
  for (const g of extracted.groups) {
    const record = byKey.get(canonical(g.key));
    if (!record || record.existence !== 'confirmed' || !record.url || !record.title || !record.authors?.length || !record.evidence?.length) {
      throw new Error(`Reference has not been verified: ${g.key} ${g.title}`);
    }
  }
  const urlKeys = new Map(), anchorKeys = new Map();
  const addURL = (url, key) => {
    const id = paperIdentity(url), set = urlKeys.get(id) || new Set();
    set.add(canonical(key)); urlKeys.set(id, set);
  };
  for (const r of catalog.records) for (const u of [r.url, ...(r.alternateUrls || [])]) if(u) addURL(u, r.key);
  // Original URLs are retained as aliases only for a verified work. Explicit
  // rejected aliases prevent a source typo from redirecting an unrelated paper.
  for (const e of extracted.entries) {
    for (const l of e.links) if (!catalog.rejectedUrls?.includes(l.url) && !/github\.com|^code$|^dataset$/i.test(l.url+' '+l.label)) addURL(l.url, e.key);
    for (const a of e.anchors) anchorKeys.set(`${e.page}#${a}`, canonical(e.key));
  }
  for (const [url,key] of Object.entries(catalog.urlAliases || {})) addURL(url,key);
  const citationReport = {linked: [], ambiguous: [], preservedText: [], numbered: [], narrative: [], removedCitationParentheses: 0, numbering: Object.fromEntries(numbers)};

  function transformBody(raw, page) {
    const lines = raw.split('\n');
    const headingCounts=new Map();
    for (const section of extracted.sections.filter(s => s.page === page)) {
      const entries = extracted.entries.filter(e => section.entries.includes(e.occurrence));
      const aliases = entries.flatMap(e => e.anchors.map(a => `<span id="${esc(a)}" data-reference-target="references.html#${canonical(e.key)}"></span>`)).join('');
      const base=section.title.toLowerCase(), count=(headingCounts.get(base)||0)+1;
      headingCounts.set(base,count);
      const headingID=base+(count>1?`-${count}`:'');
      // Preserve old incoming URLs without retaining a local bibliography block.
      lines[section.start] = `\n<div hidden><span id="${headingID}" data-reference-target="references.html"></span>${aliases}</div>\n`;
      for (let line = section.start + 1; line < section.end; line++) lines[line] = '';
    }
    let body=lines.join('\n');
    if(page==='applications')body=body.replace('(**OUR PREPRINT**)','(**OUR PREPRINT** <a class="citation-todo" href="references.html#clear-todo" title="Add the CLEAR preprint reference and source link">TODO</a>)');
    return body;
  }

  function lookupLink(href, page) {
    if (href.startsWith('#')) return anchorKeys.get(`${page}#${decodeURIComponent(href.slice(1))}`);
    const set = urlKeys.get(paperIdentity(href));
    return set?.size === 1 ? [...set][0] : undefined;
  }

  function linkCitations(tokens, page) {
    const localEntries = extracted.entries.filter(e => e.page === page);
    const entries = [...extracted.entries.filter(e => e.page !== page), ...localEntries];
    const aliases = new Map();
    const localAliases = new Map();
    const add = (s, key) => {
      const set = aliases.get(s) || new Set(); set.add(canonical(key)); aliases.set(s, set);
    };
    // Author/year labels come from the unmodified source, including its year
    // suffixes. A metadata correction never silently changes a citation label.
    for (const e of entries) {
      const names = surnames(e.authors);
      if (!names[0]) continue;
      const authorForms = [names[0]];
      if (names.length > 1 || /et al/.test(e.authors)) authorForms.push(`${names[0]} et al.`, `${names[0]} et al`);
      if (names.length === 2) authorForms.push(`${names[0]} and ${names[1]}`, `${names[0]} & ${names[1]}`);
      if (names.length === 3) authorForms.push(`${names[0]}, ${names[1]}, and ${names[2]}`, `${names[0]}, ${names[1]} and ${names[2]}`);
      for (const a of authorForms) for (const s of [`${a}, ${e.yearLabel}`, `${a} (${e.yearLabel})`, `${a} (${e.yearLabel}`, `${a} ${e.yearLabel}`]) {
        add(s, e.key);
        if(e.page===page){const set=localAliases.get(s)||new Set();set.add(canonical(e.key));localAliases.set(s,set);}
      }
    }
    for(const [s,set] of localAliases)aliases.set(s,set);
    const exactOverrides = catalog.citationAliases?.[page] || {};
    for (const [s,key] of Object.entries(exactOverrides)) aliases.set(s,new Set([canonical(key)]));
    const labels = [...aliases.keys()].sort((a,b)=>b.length-a.length);
    const pattern = labels.length ? new RegExp(`(?<![\\p{L}])(?:${labels.map(escapeRE).join('|')})(?![\\p{L}\\d])`, 'gu') : null;
    const labelKey=label=>{
      const keys=new Set([...label.matchAll(pattern)].flatMap(m=>[...aliases.get(m[0])]));
      return keys.size===1?[...keys][0]:undefined;
    };
    let linked = 0;
    for (const inline of tokens.filter(t => t.type === 'inline' && t.children)) {
      const explicitKeys=new Set(inline.children.filter(t=>t.type==='link_open').map(t=>lookupLink(t.attrGet('href')||'',page)).filter(Boolean));
      const original = inline.children.filter(t=>t.type!=='link_open'&&t.type!=='link_close').map(t=>`${t.type}:${t.content}`).join('|');
      const visibleBefore = inline.children.filter(t=>t.type!=='link_open'&&t.type!=='link_close').map(t=>t.content).join('');
      const next = []; let depth = 0;
      for (const [tokenIndex,token] of inline.children.entries()) {
        if (token.type === 'link_open') {
          depth++;
          const end=inline.children.findIndex((t,i)=>i>tokenIndex&&t.type==='link_close');
          const label=inline.children.slice(tokenIndex+1,end).map(t=>t.content).join('');
          const key = lookupLink(token.attrGet('href') || '', page)||labelKey(label);
          if (key) {
            token.attrSet('href', `references.html#${key}`); token.attrJoin('class', 'citation');
            token.attrSet('title', byKey.get(key).title);
            citationReport.linked.push({page,key,kind:'existing-link'}); linked++;
          }
        } else if (token.type === 'link_close') depth--;
        if (token.type !== 'text' || depth || !pattern) { next.push(token); continue; }
        let offset = 0;
        const appendText = value => { if(value) { const t = new token.constructor('text','',0); t.content=value; next.push(t); } };
        for (const match of token.content.matchAll(pattern)) {
          if(match.index<offset)continue;
          let set = aliases.get(match[0]);
          if(set.size>1){const explicit=[...set].filter(key=>explicitKeys.has(key));if(explicit.length===1)set=new Set(explicit);}
          if (set.size !== 1) { citationReport.ambiguous.push({page,label:match[0],keys:[...set],context:inline.content}); continue; }
          const key = [...set][0];
          let label=match[0];
          if(label.includes('(')&&!label.includes(')')){
            const rest=token.content.slice(match.index+label.length),end=rest.indexOf(')');
            if(/^[,/]/.test(rest)&&end>=0&&end<160)label+=rest.slice(0,end+1);
          }
          appendText(token.content.slice(offset,match.index));
          const open = new token.constructor('link_open','a',1);
          open.attrs = [['href',`references.html#${key}`],['class','citation'],['title',byKey.get(key).title]];
          next.push(open); appendText(label); next.push(new token.constructor('link_close','a',-1));
          offset = match.index + label.length;
          citationReport.linked.push({page,key,label,kind:'author-year'}); linked++;
        }
        appendText(token.content.slice(offset));
      }
      inline.children = next;
      const visibleAfter = next.filter(t=>t.type!=='link_open'&&t.type!=='link_close').map(t=>t.content).join('');
      if (visibleBefore !== visibleAfter) throw new Error(`Citation overlay changed prose in ${page}: ${original}`);
    }
    citationReport.preservedText.push({page,verified:true,links:linked});
    numberCitations(tokens,page);
  }

  // Sentence-initial narrative citations retain the source's author wording.
  // Other citations use [n], without a surrounding citation-only parenthesis.
  function numberCitations(tokens,page) {
    for(const [blockIndex,inline] of tokens.entries()){
      if(inline.type!=='inline'||!inline.children)continue;
      const children=inline.children, spans=[];
      for(let i=0;i<children.length;i++){
        const open=children[i];
        if(open.type!=='link_open'||!open.attrGet('class')?.split(' ').includes('citation'))continue;
        const end=children.findIndex((t,j)=>j>i&&t.type==='link_close');
        const key=open.attrGet('href').split('#')[1];
        const label=children.slice(i+1,end).map(t=>t.content).join('').trim();
        const numeric=/^(?:\\?\[)?\d+(?:\\?\])?$/.test(label);
        const prefix=children.slice(0,i).filter(t=>!t.type.startsWith('math')).map(t=>t.content).join('').trimEnd();
        const sentenceStart=!prefix.trim()||/[.!?]["”’]?\s*$/.test(prefix);
        const insideParenthesis=(prefix.match(/\(/g)||[]).length>(prefix.match(/\)/g)||[]).length;
        let author=tokens[blockIndex-1]?.type==='paragraph_open'&&sentenceStart&&!insideParenthesis?label.match(authorStartPattern)?.[1]:undefined;
        if(author?.includes(','))author=author.split(',')[0]+' et al.';
        spans.push({start:i,end,key,numeric,label,author});i=end;
      }
      // A source may use "Author (year) [local-number]". Retain its prose,
      // replace the local number, and avoid adding a second identical marker.
      for(const [i,s] of spans.entries()){
        if(s.label==='https://hackmd.io/team/moosurveyXu'&&s.key==='r004'){
          const after=children[s.end+1];
          if(after?.type==='text')after.content=after.content.replace(/^\s*et al\., 2025/,'');
        }
        const following=spans[i+1];
        const gap=following?children.slice(s.end+1,following.start).map(t=>t.content).join(''):'';
        s.alreadyNumbered=!s.numeric&&following?.numeric&&following.key===s.key&&gap.length<160&&!/[.!?]\s+[A-Z]/.test(gap);
        if(s.numeric){
          const before=children[s.start-1],after=children[s.end+1];
          if(before?.type==='text'&&after?.type==='text'&&/\[$/.test(before.content)&&/^\]/.test(after.content)){
            before.content=before.content.slice(0,-1);after.content=after.content.slice(1);
          }
        }
      }
      const byStart=new Map(spans.map(s=>[s.start,s]));
      const next=[];
      for(let i=0;i<children.length;i++){
        const s=byStart.get(i);
        if(!s){next.push(children[i]);continue;}
        const open=children[i],close=children[s.end];
        let label=s.numeric?'':withoutAuthorYear(s.label);
        if(s.author){
          label=s.author+(label?`, ${label}`:'');
          citationReport.narrative.push({page,key:s.key,author:s.author,number:numbers.get(s.key)});
        }
        if(label){
          const text=new open.constructor('text','',0);text.content=label;
          if(s.author){const tag=new open.constructor('html_inline','',0);tag.content='<span class="narrative-author">';next.push(tag,text);const end=new open.constructor('html_inline','',0);end.content='</span>';next.push(end);}
          else next.push(text);
        }
        if(!s.alreadyNumbered){
          const marker=new open.constructor('text','',0);marker.content=`[${numbers.get(s.key)}]`;marker.meta={citation:true};
          if(label){const space=new open.constructor('text','',0);space.content=' ';next.push(space);}
          open.attrSet('data-reference-number',String(numbers.get(s.key)));
          open.attrSet('aria-label',`Reference ${numbers.get(s.key)}: ${byKey.get(s.key).title}`);
          next.push(open,marker,close);
          citationReport.numbered.push({page,key:s.key,number:numbers.get(s.key)});
        }
        i=s.end;
      }
      removeCitationParentheses(next);
      inline.children=next;
    }
  }

  function removeCitationParentheses(children) {
    let text='';const owners=[];
    for(const [index,t] of children.entries()){
      const value=t.type==='text'?t.content:t.content?'\ufffc':'';
      for(let offset=0;offset<value.length;offset++)owners.push({index,offset,citation:!!t.meta?.citation});
      text+=value;
    }
    const edits=new Map();
    for(const match of text.matchAll(/\(\s*\[\d+\](?:\s*[,;]?\s*\[\d+\])*\s*\)/g)){
      if(!owners.slice(match.index,match.index+match[0].length).some(o=>o.citation))continue;
      for(const position of [match.index,match.index+match[0].length-1]){
        const {index,offset}=owners[position];
        const list=edits.get(index)||[];
        const replacement=position===match.index&&position>0&&/[\p{L}\p{N}]/u.test(text[position-1])?' ':'';
        list.push({offset,replacement});edits.set(index,list);
      }
      citationReport.removedCitationParentheses++;
    }
    for(const [index,list] of edits)for(const {offset,replacement} of list.sort((a,b)=>b.offset-a.offset))children[index].content=children[index].content.slice(0,offset)+replacement+children[index].content.slice(offset+1);
  }

  function bibliography() {
    return `<h1 id="references">References</h1><ol class="reference-list">${sorted.map(r=>`<li class="reference-entry" id="${r.key}" value="${numbers.get(r.key)}" tabindex="-1">${esc(r.authors.join(', '))} (${esc(r.year)}). <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a>.${r.venue?` <em>${esc(r.venue)}</em>.`:''}</li>`).join('')}</ol><p class="reference-pending" id="clear-todo" tabindex="-1">CLEAR — <span class="citation-todo">TODO: add preprint reference and link.</span></p>`;
  }
  return {transformBody,linkCitations,bibliography,records,citationReport,extracted};
}
