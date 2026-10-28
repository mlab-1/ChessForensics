/* Chess Forensics — Single-page UI */

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

function esc(s) { return String(s??'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function isNum(v) { return typeof v === 'number' && Number.isFinite(v); }
function fmt(v, d=2) { return isNum(v) ? v.toFixed(d) : '--'; }
function pct(v, d=1) { return isNum(v) ? (v*100).toFixed(d)+'%' : '--'; }
function signed(v, d=1) { return isNum(v) ? (v>=0?'+':'')+v.toFixed(d) : '--'; }
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

// ================================================================ SCAN BOARD CYCLING
const PIECE_CH={'K':'\u2654','Q':'\u2655','R':'\u2656','B':'\u2657','N':'\u2658','P':'\u2659',
                'k':'\u265A','q':'\u265B','r':'\u265C','b':'\u265D','n':'\u265E','p':'\u265F'};
const SCAN_FENS=[
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR',
  'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R',
  'r1bqkb1r/pppppppp/2n2n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR',
  'rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R',
  'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R',
  'rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR',
  'r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R',
  'r2q1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1BP2/PP4PP/R2QKB1R',
  'r1bq1rk1/pp2bppp/2nppn2/8/2BNP3/2N1B3/PPP2PPP/R2QK2R',
  'r2qr1k1/pp1nbppp/2p2n2/3p4/3P4/1QN1PN2/PP3PPP/R1B2RK1',
  '2rq1rk1/pp1bppbp/2np1np1/8/2PNP3/2N1BP2/PP4PP/R2QKB1R',
  'r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/N1P2N1P/PP3PP1/R1BQRK2',
  'r4rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P5PP/R2QKB1R',
  'rnb1k2r/ppq1ppbp/5np1/2pp4/2P5/1PN1PN2/P2PBPPP/R1BQK2R',
  '2r2rk1/1bqnbppp/pp1ppn2/8/P1PNP3/1PN1BP2/6PP/R2QKB1R',
  'r1bqk2r/pppp1Npp/2n2n2/2b1p3/2B1P3/8/PPPP1PPP/RNBQK2R',
  'r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R',
  '6k1/pp3ppp/2p2n2/3r4/8/2N2P2/PPP3PP/R5K1',
  'r4rk1/pp3ppp/2n5/8/2B5/2b2N2/P4PPP/R4RK1',
  '8/5pk1/6p1/4P3/p1p5/P4K1P/1P4P1/8',
  '3r2k1/pp3pp1/2p4p/8/3Pb3/4PN2/PP3PPP/3R2K1',
  'r3kb1r/pbqn1ppp/1p2pn2/2p5/2P5/1PN2NP1/P2PPPBP/R1BQ1RK1',
  'rn1qkbnr/ppp2ppp/4p3/3pPb2/3P4/8/PPP2PPP/RNBQKBNR',
  'r1b1r1k1/pp1nqppp/2p1p3/8/2BP4/2N1PN2/PP3PPP/2RQ1RK1',
  '1rb2rk1/p1p1qppp/2pp1n2/4p3/2P1P3/1P3N2/PB1NQPPP/R4RK1',
  '5rk1/1p3pp1/p1np3p/4p3/2PqP3/3P1N1P/PP3PP1/R2Q1RK1',
  'r2q1rk1/1pp2ppp/p1np1n2/2b1p1B1/2B1P3/3P1N2/PPP2PPP/R2QR1K1',
  '4rrk1/1ppq1p1p/p2p1bp1/8/2PnP3/2NB4/PP2QPPP/R4RK1',
  'r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R',
  '2r3k1/1p2qppp/p3rn2/8/1b1NR3/1P4P1/PB2QPKP/3R4',
];
function fenToGrid(fen){
  const rows=fen.split('/'),grid=[];
  for(const row of rows){
    const cells=[];
    for(const ch of row){
      if(ch>='1'&&ch<='8') for(let i=0;i<parseInt(ch);i++) cells.push('');
      else cells.push({ch:PIECE_CH[ch]||'',w:ch===ch.toUpperCase()});
    }
    grid.push(cells);
  }
  return grid;
}
let _scanTimer=null,_scanEl=null;
function startBoardCycling(){
  const c=$('#scan-boards');if(!c)return;
  let h='<div class="scan-board-grid">';
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
    const cls=(r+c)%2===0?'light':'dark';
    h+=`<div class="scan-cell ${cls}"></div>`;
  }
  h+='</div>';c.innerHTML=h;
  _scanEl=c.querySelector('.scan-board-grid');
  let idx=Math.floor(Math.random()*SCAN_FENS.length);
  function cycle(){
    idx=(idx+1)%SCAN_FENS.length;
    const grid=fenToGrid(SCAN_FENS[idx]);
    const cells=_scanEl.querySelectorAll('.scan-cell');
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){
      const cell=cells[r*8+c],p=grid[r]?.[c];
      cell.textContent=p?p.ch:'';
      cell.classList.toggle('w',!!p&&p.w);
      cell.classList.toggle('b',!!p&&!p.w);
    }
  }
  cycle();
  _scanTimer=setInterval(cycle,130);
}
function stopBoardCycling(){
  if(_scanTimer){clearInterval(_scanTimer);_scanTimer=null}
  const c=$('#scan-boards');if(c)c.innerHTML='';
}

function animateValue(el, end, dur=1200, dec=1) {
  const t0 = performance.now();
  const step = t => {
    const p = Math.min((t-t0)/dur, 1);
    const e = 1 - Math.pow(1-p, 3);
    el.textContent = (end * e).toFixed(dec);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ================================================================ STARS
function generateStars() {
  const cosmos = $('#cosmos');
  for (let i = 0; i < 200; i++) {
    const s = document.createElement('div');
    s.className = 'star';
    const size = Math.random() < 0.1 ? 2 : 1;
    s.style.cssText = `
      width:${size}px;height:${size}px;
      left:${Math.random()*100}%;top:${Math.random()*100}%;
      --dur:${2+Math.random()*5}s;--lo:${0.1+Math.random()*0.3};--hi:${0.6+Math.random()*0.4};
      animation-delay:${-Math.random()*5}s;
    `;
    cosmos.appendChild(s);
  }
}
generateStars();

// ================================================================ PLOTLY THEME
const PF = { color:'#6b8095', family:'Outfit,sans-serif', size:10.5 };
const AX = { gridcolor:'rgba(0,180,255,0.06)', zerolinecolor:'rgba(255,255,255,0.1)', linecolor:'rgba(255,255,255,0.06)', tickfont:PF, titlefont:{...PF,size:10.5} };
function mklay(x={}) {
  return {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:PF,
    margin:{l:50,r:20,t:10,b:44},
    hoverlabel:{bgcolor:'rgba(8,20,35,0.9)',bordercolor:'rgba(0,180,255,0.3)',font:{...PF,color:'#e0eaf0',size:11}},
    legend:{orientation:'h',y:-0.22,font:PF},
    ...x, xaxis:{...AX,...(x.xaxis||{})}, yaxis:{...AX,...(x.yaxis||{})},
  };
}
const PC = {displayModeBar:false,responsive:true};
function pl(n,d,l){if(n)Plotly.newPlot(n,d,l,PC)}

const C = { cyan:'#a0b4f0',blue:'#7b8fd8',deepblue:'#5060b0',teal:'#6aada0',pink:'#d48aaf',gold:'#d4b060',red:'#d45060',emerald:'#50c090',fg:'#e8e6f0',fg2:'#9a96b0' };

// ================================================================ API
async function api(path, body) {
  const r = await fetch(path, body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let p; try{p=await r.json()}catch{p=null}
  if(!r.ok) throw new Error((p&&p.detail)||`${r.status}`);
  return p;
}
function notice(kind, html, autoHide=0) {
  const d = document.createElement('div'); d.className = `notice ${kind}`;
  d.innerHTML = html; $('#msgs').appendChild(d);
  if (autoHide > 0) setTimeout(() => { d.style.transition='opacity 0.4s'; d.style.opacity='0'; setTimeout(()=>d.remove(), 400); }, autoHide);
}
function clearMsgs() { $('#msgs').innerHTML = ''; }

// ================================================================ STATE
const state = { mode:'paste', analysisType:'single', scanSource:'chesscom', candidates:[], report:null, scanReport:null, jobId:null };

function setAnalysisType(t) {
  state.analysisType = t;
  $$('#analysis-type button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.type===t)));
  $('#single-inputs').hidden = t !== 'single';
  $('#scan-inputs').hidden = t !== 'scan';
  $('#deep').closest('label').hidden = t === 'scan'; // scan never uses deep pass
  $('#run').textContent = t === 'scan' ? 'Scan Player' : 'Analyse';
}
const SPEED_OPTS = {
  chesscom: [['','All'],['rapid','Rapid'],['blitz','Blitz'],['bullet','Bullet'],['daily','Daily']],
  lichess:  [['','All'],['rapid,classical','Rapid+Classical'],['rapid','Rapid'],['blitz','Blitz'],['bullet','Bullet'],['classical','Classical']],
};
function setScanSource(s) {
  state.scanSource = s;
  $$('#scan-source button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.src===s)));
  const sel = $('#scan-speed');
  if (sel) {
    const opts = SPEED_OPTS[s] || SPEED_OPTS.chesscom;
    sel.innerHTML = opts.map(([v,l]) => `<option value="${v}"${v==='rapid'?' selected':''}>${l}</option>`).join('');
  }
}

function setMode(m) {
  state.mode = m;
  $$('#modes button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode===m)));
  $$('.mode-body').forEach(d => { d.hidden = d.dataset.for!==m; });
  state.candidates = []; renderPicker();
}
function renderPicker() {
  const w=$('#pickwrap'),s=$('#pick');
  if(!state.candidates.length){w.hidden=true;s.innerHTML='';return}
  s.innerHTML=state.candidates.map((c,i)=>`<option value="${i}">${esc(c.label)}</option>`).join('');
  w.hidden=false;
}
function selection() {
  if (state.candidates.length>0) { const i=parseInt($('#pick').value||'0',10); return {pgn:state.candidates[i].pgn,index:state.candidates[i].index||0}; }
  const t=$('#pgn').value.trim(); return t?{pgn:t,index:0}:null;
}
async function doImport(mode,value,count,speed='') {
  clearMsgs(); const btn=document.activeElement; if(btn)btn.disabled=true;
  try{ const out=await api('/api/import',{mode,value,count,speed}); state.candidates=out.games; renderPicker();
    notice('info',`Fetched <strong>${out.games.length}</strong> game(s).`);
  }catch(e){notice('err',`<strong>Import failed.</strong> ${esc(e.message)}`);}
  finally{if(btn)btn.disabled=false}
}
async function inspectPasted() {
  const t=$('#pgn').value.trim();
  if(!t){state.candidates=[];renderPicker();return}
  try{const o=await api('/api/pgn/inspect',{pgn:t}); state.candidates=o.count>1?o.labels.map((l,i)=>({label:l,pgn:t,index:i})):[];renderPicker();}
  catch{state.candidates=[];renderPicker()}
}

// ================================================================ ANALYSIS
function setProgress(done,total,label) {
  const f=total>0?clamp(done/total,0,1):0;
  $('#progfill').style.width=`${(f*100).toFixed(1)}%`;
  $('#progtext').innerHTML=`<span class="spin"></span> ${esc(label)}${total>0?` (${done}/${total})`:''}`;
}

async function runAnalysis() {
  if (state.analysisType === 'scan') return runScan();
  const sel=selection();
  if(!sel){clearMsgs();notice('err','Paste a PGN or import a game first.');return}
  let idx=sel.index;
  if(state.candidates.length>0){const c=state.candidates[parseInt($('#pick').value||'0',10)];if(typeof c.index==='number')idx=c.index}
  clearMsgs();
  $('#run').disabled=true;$('#run').textContent='Analysing...';
  $('#progwrap').hidden=false;startBoardCycling();setProgress(0,0,'starting engines');
  $('#input-overlay').classList.add('hidden');
  const cancelBtnA=$('#cancel-scan');if(cancelBtnA)cancelBtnA.hidden=false;
  try{
    const{job_id}=await api('/api/analyse',{pgn:sel.pgn,game_index:idx,prior:parseFloat($('#prior').value)||0.01,deep_pass:$('#deep').checked,pool_size:parseInt($('#pool').value,10)||10});
    state.jobId=job_id; await followJob(job_id);
  }catch(e){if(e.message)notice('err',`<strong>Failed.</strong> ${esc(e.message)}`);$('#input-overlay').classList.remove('hidden')}
  finally{$('#run').disabled=false;$('#run').textContent='Analyse';stopBoardCycling();$('#progwrap').hidden=true;const cb=$('#cancel-scan');if(cb)cb.hidden=true}
}

async function runScan() {
  const user=$('#scan-user').value.trim();
  if(!user){clearMsgs();notice('err','Enter a username.');return}
  clearMsgs();
  $('#run').disabled=true;$('#run').textContent='Scanning...';
  $('#progwrap').hidden=false;startBoardCycling();setProgress(0,0,'fetching games');
  $('#input-overlay').classList.add('hidden');
  const cancelBtn=$('#cancel-scan');if(cancelBtn)cancelBtn.hidden=false;
  try{
    const quick=$('#scan-quick')&&$('#scan-quick').checked;
    const{job_id}=await api('/api/scan',{mode:state.scanSource,username:user,count:parseInt($('#scan-count').value,10)||20,prior:parseFloat($('#prior').value)||0.01,pool_size:parseInt($('#pool').value,10)||10,speed:$('#scan-speed').value,quick});
    state.jobId=job_id; await followScanJob(job_id);
  }catch(e){if(e.message)notice('err',`<strong>Failed.</strong> ${esc(e.message)}`);$('#input-overlay').classList.remove('hidden')}
  finally{$('#run').disabled=false;$('#run').textContent='Scan Player';stopBoardCycling();$('#progwrap').hidden=true;if(cancelBtn)cancelBtn.hidden=true}
}

function followScanJob(id) {
  return new Promise((resolve,reject)=>{
    const es=new EventSource(`/api/jobs/${id}/events`);
    es.onmessage=async ev=>{
      const m=JSON.parse(ev.data);
      if(m.state==='done'){es.close();try{const r=await api(`/api/jobs/${id}`);state.scanReport=r.report;renderScanReport();resolve()}catch(e){reject(e)}return}
      if(m.state==='cancelled'){es.close();notice('info','Scan cancelled.',3000);resolve();return}
      if(m.state==='error'){es.close();let tr;try{tr=(await api(`/api/jobs/${id}`)).trace}catch{}
        notice('err',`<strong>Failed.</strong> ${esc(m.error)}${tr?`<details><summary>traceback</summary><pre style="white-space:pre-wrap;color:var(--fg-3);font-size:11px">${esc(tr)}</pre></details>`:''}`);
        reject(new Error(''));return}
      // scan progress: game_done/game_total
      const gd=m.game_done||0, gt=m.game_total||0;
      const md=m.move_done||0, mt=m.move_total||0;
      const f=gt>0?clamp(gd/gt,0,1):0;
      $('#progfill').style.width=`${(f*100).toFixed(1)}%`;
      $('#progtext').innerHTML=`<span class="spin"></span> ${esc(m.label||'working')}${gt>0?` (game ${gd+1}/${gt})`:''}${mt>0?` move ${md}/${mt}`:''}`;
    };
    es.onerror=()=>{if(es.readyState===EventSource.CLOSED)reject(new Error('lost connection'))};
  });
}

function followJob(id) {
  return new Promise((resolve,reject)=>{
    const es=new EventSource(`/api/jobs/${id}/events`);
    es.onmessage=async ev=>{
      const m=JSON.parse(ev.data);
      if(m.state==='done'){es.close();try{state.report=(await api(`/api/jobs/${id}`)).report;renderReport();resolve()}catch(e){reject(e)}return}
      if(m.state==='cancelled'){es.close();notice('info','Analysis cancelled.',3000);resolve();return}
      if(m.state==='error'){es.close();let tr;try{tr=(await api(`/api/jobs/${id}`)).trace}catch{}
        notice('err',`<strong>Failed.</strong> ${esc(m.error)}${tr?`<details><summary>traceback</summary><pre style="white-space:pre-wrap;color:var(--fg-3);font-size:11px">${esc(tr)}</pre></details>`:''}`);
        reject(new Error(''));return}
      setProgress(m.done||0,m.total||0,m.label||'working');
    };
    es.onerror=()=>{if(es.readyState===EventSource.CLOSED)reject(new Error('lost connection'))};
  });
}

// Book detection is now done server-side via the Lichess opening explorer.

// ================================================================ BOARD
let boardLib=null;
function loadBoardLib(){
  if(boardLib)return boardLib;
  const b='https://cdn.jsdelivr.net/npm/cm-chessboard@8.7.5/src/';
  boardLib=Promise.all([import(`${b}Chessboard.js`),import(`${b}extensions/markers/Markers.js`)]).then(([cb,mk])=>({Chessboard:cb.Chessboard,BORDER_TYPE:cb.BORDER_TYPE,COLOR:cb.COLOR,Markers:mk.Markers,MARKER_TYPE:mk.MARKER_TYPE}));
  return boardLib;
}
const rp={ply:0,board:null,lib:null,flipped:false};

function setPly(ply){
  const rep=state.report,moves=rep.moves;
  ply=clamp(Math.round(ply),0,moves.length-1);rp.ply=ply;const m=moves[ply];
  const sl=$('#plyslider');if(sl)sl.value=String(ply);

  // highlight active move in list
  const ml=$('#movelist');
  if(ml){
    for(const c of ml.querySelectorAll('.ml-move.active')) c.classList.remove('active');
    const active=ml.querySelector(`.ml-move[data-idx="${ply}"]`);
    if(active){active.classList.add('active');active.scrollIntoView({block:'nearest',behavior:'smooth'})}
  }

  // board position
  if(rp.board&&rp.lib){const lib=rp.lib;
    let fenAfter=moves[ply+1]?moves[ply+1].fen_before:null;
    if(!fenAfter&&rep.final_fen) fenAfter=rep.final_fen;
    rp.board.setPosition(fenAfter||m.fen_before,false);
    rp.board.removeMarkers();
    if(m.uci&&m.uci.length>=4){const f=m.uci.slice(0,2),t=m.uci.slice(2,4);rp.board.addMarker(lib.MARKER_TYPE.frame,f);rp.board.addMarker(lib.MARKER_TYPE.frame,t)}
  }

  // eval bar — show eval of the position AFTER the move (matches the board)
  const evBar=$('#eval-fill'),evTxt=$('#eval-text');
  if(evBar&&evTxt&&state._evalByPly){
    const ev=state._evalByPly[ply+1]??state._evalByPly[ply];
    if(isNum(ev)){
      const wPct=clamp(50+ev*10,2,98);
      evBar.style.height=`${wPct}%`;
      if(Math.abs(ev)>10){
        evTxt.textContent='';
      }else{
        evTxt.textContent=ev>=0?`+${ev.toFixed(1)}`:ev.toFixed(1);
        evTxt.style.bottom=wPct>50?`${wPct-8}%`:`${wPct+2}%`;
        evTxt.style.color=wPct>50?'#333':'#ddd';
      }
    }
  }
}

// ================================================================ RENDER REPORT (single page)
function renderReport() {
  const app=$('#app');
  app.classList.add('has-results');
  const n=$('#report');
  n.hidden=false;
  clearMsgs();
  const rep=state.report;
  if(rep.uncalibrated) notice('err','<strong>Uncalibrated.</strong> Using shipped defaults.');
  else if(rep.provisional) notice('warn',`<strong>Provisional.</strong> ${esc(rep.baseline_description)}`);

  const gi=rep.game_info, pl_=rep.players, moves=rep.moves, curve=rep.eval_curve||[], recs=rep.frame_records||[];
  const ins=rep.insights||{};
  const sub=[gi.event,gi.date,gi.opening,gi.speed].filter(Boolean).map(esc).join(' / ');

  // Build eval lookup: ply -> eval_white (fill gaps with last known value)
  // Includes moves.length (position after the final move) so the eval bar
  // for the last ply can show the post-move eval.
  const evalMap={};
  for(const e of curve) if(isNum(e.eval_white)) evalMap[e.ply]=e.eval_white;
  state._evalByPly={};
  let lastEv=0;
  for(let i=0;i<=moves.length;i++){
    if(evalMap[i]!==undefined) lastEv=evalMap[i];
    state._evalByPly[i]=lastEv;
  }

  // Build simple move list (no badges)
  let mlHtml='';
  for(let i=0;i<moves.length;i+=2){
    const w=moves[i], b=moves[i+1]||null;
    mlHtml+=`<div class="ml-row">
      <span class="ml-num">${w.move_number}.</span>
      <span class="ml-move" data-idx="${i}">${esc(w.san)}</span>
      ${b?`<span class="ml-move" data-idx="${i+1}">${esc(b.san)}</span>`:'<span class="ml-move"></span>'}
    </div>`;
  }

  // Eval chart data
  const evalPlies=curve.map(r=>r.ply), evalVals=curve.map(r=>clamp(r.eval_white,-10,10));

  n.innerHTML=`
    <!-- GAME HEADER -->
    <div id="gamehead">
      <div class="names">${esc(gi.white)}<span class="elo">${esc(gi.white_elo||'?')}</span>
        <span class="vs">VS</span>
        ${esc(gi.black)}<span class="elo">${esc(gi.black_elo||'?')}</span></div>
      <div class="meta">${sub}${gi.result?` &mdash; ${esc(gi.result)}`:''}</div>
    </div>

    <!-- BOARD SECTION -->
    <div class="board-section">
      <div class="board-center">
        <div id="board-area">
          <div id="eval-bar"><div id="eval-fill"></div><div id="eval-text">0.0</div></div>
          <div id="board-wrap"><div id="board"></div></div>
        </div>
        <div class="replay-controls">
          <button id="rwd">&#9198;</button><button id="prev">&#9664;</button>
          <button id="next">&#9654;</button><button id="ffwd">&#9197;</button>
          <button id="flip">&#8635;</button>
        </div>
        <input type="range" id="plyslider" min="0" max="${moves.length-1}" value="0">
      </div>
      <div class="ml-wrap">
        <div class="ml-header">Moves</div>
        <div class="ml-body" id="movelist">${mlHtml}</div>
      </div>
    </div>

    <!-- EVAL CHART -->
    <div class="card fade-in" style="margin-top:24px">
      <h3>Evaluation</h3>
      <p class="sub">How the position changed throughout the game</p>
      <div class="chart" id="evalchart"></div>
    </div>

    <!-- VERDICT CARDS -->
    <div class="grid2" style="margin-top:24px" id="verdict-cards">
    ${['white','black'].map(color=>{
      const p=pl_[color];
      const pp=(ins.per_player||{})[color]||{};
      const acc=isNum(pp.accuracy)?pp.accuracy:null;
      const v=p.verdict;
      const insufficient=p.insufficient||!v;

      let lightColor, lightLabel, lightClass;
      if(insufficient){ lightColor='var(--fg-3)'; lightLabel='Insufficient data'; lightClass='light-grey'; }
      else if(v.posterior>=0.5){ lightColor='var(--red)'; lightLabel='Highly unusual play'; lightClass='light-red'; }
      else if(v.posterior>=0.05){ lightColor='var(--gold)'; lightLabel='Some unusual patterns'; lightClass='light-yellow'; }
      else { lightColor='var(--emerald)'; lightLabel='Normal play'; lightClass='light-green'; }

      const accCol=acc===null?'var(--fg-3)':acc>=90?'var(--emerald)':acc>=70?'var(--cyan)':acc>=50?'var(--gold)':'var(--red)';

      return `<div class="card verdict-card fade-in">
        <h3>${esc(p.name)} <span class="qual">(${color})</span></h3>
        ${p.elo?`<div class="verdict-elo">Rated ${esc(p.elo)}</div>`:''}

        <div class="traffic-light ${lightClass}">
          <div class="light-dot" style="background:${lightColor};box-shadow:0 0 20px ${lightColor}"></div>
          <div class="light-label">${lightLabel}</div>
        </div>

        ${acc!==null?`
          <div class="acc-ring-wrap">
            <svg class="acc-ring" viewBox="0 0 200 200">
              <circle class="acc-bg" cx="100" cy="100" r="80" stroke-dasharray="${2*Math.PI*80}" transform="rotate(-90 100 100)"/>
              <circle class="acc-val" cx="100" cy="100" r="80" stroke="${accCol}" stroke-dasharray="${2*Math.PI*80*clamp(acc/100,0,1)} ${2*Math.PI*80*(1-clamp(acc/100,0,1))}" transform="rotate(-90 100 100)" style="filter:url(#glow)"/>
              <text class="acc-num" x="100" y="92" id="acc-${color}">${acc.toFixed(1)}</text>
              <text class="acc-label" x="100" y="118">ACCURACY</text>
            </svg>
          </div>`:'<p class="empty" style="margin:20px 0">Insufficient data</p>'}

        ${!insufficient&&v?`
          <div class="verdict-detail">
            <span>Engine match: <strong>${pct((p.summary||{}).t1)}</strong></span>
            <span>Expected: <strong>${p.t1_test?pct(p.t1_test.expected/((p.summary||{}).n_scoreable||1)):'--'}</strong></span>
          </div>
          ${v.interpretation?`<p class="verdict-interp">${esc(v.interpretation)}</p>`:''}
        `:''}
      </div>`;
    }).join('')}
    </div>

    <!-- ENGINE AGREEMENT -->
    <div class="card fade-in" style="margin-top:24px">
      <h3>Engine Agreement</h3>
      <p class="sub">How often each player matched the engine's top choice vs what's normal for their rating</p>
      <div id="agreement-bars">
      ${['white','black'].map(color=>{
        const p=pl_[color], ns=(p.summary||{}).n_scoreable||1;
        const obs=(p.summary||{}).t1||0;
        const t1=p.t1_test;
        const obsPct=obs*100, expPct=t1?(t1.expected/ns*100):null;
        return `<div class="agree-row">
          <div class="agree-name">${esc(p.name)}</div>
          <div class="agree-track">
            <div class="agree-fill" style="width:${clamp(obsPct,0,100)}%"></div>
            ${expPct!==null?`<div class="agree-marker" style="left:${clamp(expPct,0,100)}%" title="Expected for rating: ${expPct.toFixed(1)}%"></div>`:''}
          </div>
          <div class="agree-pct">${obsPct.toFixed(1)}%</div>
        </div>`;
      }).join('')}
      </div>
      <div class="agree-legend"><span class="agree-leg-fill"></span> Actual <span class="agree-leg-marker"></span> Expected for rating</div>
    </div>

    <!-- MOVE RANK DISTRIBUTION -->
    <h2 class="sec">Move Rank Distribution</h2>
    <p class="secsub">How often did each player find the engine's top choice?</p>
    <div class="grid2">
    ${['white','black'].map(color=>{
      return `<div class="card fade-in">
        <h3>${esc(pl_[color].name)} <span class="qual">(${color})</span></h3>
        <div class="chart short" id="rank-dist-${color}"></div>
      </div>`;
    }).join('')}
    </div>

    <!-- THINK TIME vs MOVE QUALITY -->
    <h2 class="sec">Think Time vs Move Quality</h2>
    <p class="secsub">Moves coloured by quality (top-3 match + centipawn loss). Cheaters often show uniform timing regardless of difficulty.</p>
    <div class="grid2">
    ${['white','black'].map(color=>{
      return `<div class="card fade-in">
        <h3>${esc(pl_[color].name)} <span class="qual">(${color})</span></h3>
        <div class="chart short" id="time-quality-${color}"></div>
      </div>`;
    }).join('')}
    </div>

    <!-- CUMULATIVE EVIDENCE -->
    <h2 class="sec">Cumulative Evidence</h2>
    <p class="secsub">How the case builds move by move. Steep jumps reveal where suspicious play concentrates.</p>
    <div class="grid2">
    ${['white','black'].map(color=>{
      return `<div class="card fade-in">
        <h3>${esc(pl_[color].name)} <span class="qual">(${color})</span></h3>
        <div class="chart short" id="cumev-${color}"></div>
      </div>`;
    }).join('')}
    </div>

    <!-- ACCURACY BY GAME PHASE -->
    <h2 class="sec">Accuracy by Game Phase</h2>
    <p class="secsub">Engine match rate broken down by opening, middlegame, and endgame.</p>
    <div class="grid2">
    ${['white','black'].map(color=>{
      return `<div class="card fade-in">
        <h3>${esc(pl_[color].name)} <span class="qual">(${color})</span></h3>
        <div class="chart short" id="phase-acc-${color}"></div>
      </div>`;
    }).join('')}
    </div>


    <!-- EVIDENCE (collapsible) -->
    <details class="evidence-details fade-in" style="margin-top:24px">
      <summary class="evidence-summary">Detailed Evidence</summary>
      <div class="grid2" style="margin-top:14px">
      ${['white','black'].map(color=>{
        const p=pl_[color], v=p.verdict;
        if(!v) return `<div class="card"><h3>${esc(p.name)}</h3><p class="empty">No data.</p></div>`;
        return `<div class="card">
          <h3>${esc(p.name)}</h3>
          <div class="ev-row"><span class="ev-label">Posterior</span><span class="ev-val">${pct(v.posterior)}</span></div>
          <div class="ev-row"><span class="ev-label">Credible interval</span><span class="ev-val">${pct(v.posterior_low)} &ndash; ${pct(v.posterior_high)}</span></div>
          ${v.rows?v.rows.map(r=>`<div class="ev-row"><span class="ev-label">${esc(r.label)}</span><span class="ev-val">${isNum(r.z)?signed(r.z,2)+'z':''} ${isNum(r.decibans_adjusted)?signed(r.decibans_adjusted,1)+'db':''}</span></div>`).join(''):''}
          ${(p.caveats||[]).map(c=>`<div class="caveat">${esc(c)}</div>`).join('')}
          ${(v.caveats||[]).map(c=>`<div class="caveat">${esc(c)}</div>`).join('')}
        </div>`;
      }).join('')}
      </div>
      <div class="card" style="margin-top:14px">
        <div class="row">
          <a class="btn-ghost" href="/api/jobs/${state.jobId}/frame.csv">Download CSV</a>
          <a class="btn-ghost" href="/api/jobs/${state.jobId}/report.json">Download JSON</a>
        </div>
        <p class="sub" style="margin-top:10px">One game is almost never enough to conclude anything. This tool sees only the moves.</p>
      </div>
    </details>
  `;

  // --- Wire board ---
  for(const cell of $$('#movelist .ml-move[data-idx]')){
    cell.addEventListener('click',()=>setPly(parseInt(cell.dataset.idx,10)));
  }
  $('#plyslider').addEventListener('input',e=>setPly(parseInt(e.target.value,10)));
  $('#prev').addEventListener('click',()=>setPly(rp.ply-1));
  $('#next').addEventListener('click',()=>setPly(rp.ply+1));
  $('#rwd').addEventListener('click',()=>setPly(0));
  $('#ffwd').addEventListener('click',()=>setPly(moves.length-1));
  $('#flip').addEventListener('click',()=>{rp.flipped=!rp.flipped;if(rp.board&&rp.lib)rp.board.setOrientation(rp.flipped?rp.lib.COLOR.black:rp.lib.COLOR.white)});
  const kh=e=>{if(e.key==='ArrowLeft'){e.preventDefault();setPly(rp.ply-1)}if(e.key==='ArrowRight'){e.preventDefault();setPly(rp.ply+1)}};
  document.removeEventListener('keydown',window._rk);window._rk=kh;document.addEventListener('keydown',kh);
  loadBoardLib().then(lib=>{rp.lib=lib;rp.board=new lib.Chessboard($('#board'),{position:moves[0].fen_before,assetsUrl:'https://cdn.jsdelivr.net/npm/cm-chessboard@8.7.5/assets/',style:{cssClass:'default',borderType:lib.BORDER_TYPE.none,pieces:{file:'pieces/staunty.svg'}},extensions:[{class:lib.Markers}]});setPly(0)}).catch(()=>{$('#board').innerHTML='<p class="empty">Board failed to load.</p>'});
  setPly(0);

  // animate accuracy rings
  for(const color of['white','black']){
    const pp=(ins.per_player||{})[color]||{};
    const el=$(`#acc-${color}`);
    if(el&&isNum(pp.accuracy)) animateValue(el,pp.accuracy,1400);
  }

  // --- Eval chart ---
  if(curve.length){
    pl($('#evalchart'),[
      {x:evalPlies,y:evalVals,text:curve.map(r=>r.san),customdata:curve.map(r=>r.depth),mode:'lines',fill:'tozeroy',line:{color:C.cyan,width:1.5},fillcolor:'rgba(34,211,238,0.1)',hovertemplate:'ply %{x} %{text}: %{y:+.2f}<extra></extra>'}
    ],mklay({xaxis:{title:'move'},yaxis:{title:'eval (pawns)'},showlegend:false}));
    $('#evalchart').on('plotly_click',ev=>{if(ev.points&&ev.points.length)setPly(ev.points[0].x)});
  }

  // --- Move Rank Distribution (donut chart) ---
  for(const color of['white','black']){
    const tgt=$(`#rank-dist-${color}`);if(!tgt) continue;
    const p=pl_[color];
    const pr=recs.filter(r=>r.color===color&&r.scoreable===1);
    if(!pr.length){tgt.innerHTML='<p class="empty">No data.</p>';continue}
    const n=pr.length;
    const r1=pr.filter(r=>r.rank===1).length;
    const r2=pr.filter(r=>r.rank===2).length;
    const r3=pr.filter(r=>r.rank===3).length;
    const rOut=n-r1-r2-r3;
    const expT1=p.t1_test?(p.t1_test.expected/n*100):null;
    const centerText=`${(r1/n*100).toFixed(0)}%`;
    pl(tgt,[{
      values:[r1,r2,r3,rOut],
      labels:['#1 Best','#2','#3','Other'],
      type:'pie',hole:0.55,
      marker:{colors:[C.emerald,C.cyan,C.gold,'rgba(212,80,96,0.6)'],line:{color:'rgba(0,0,0,0.3)',width:1}},
      textinfo:'label+percent',textposition:'outside',
      textfont:{family:'Outfit',size:12,color:C.fg2},
      hovertemplate:'%{label}: %{value} moves (%{percent})<extra></extra>',
      sort:false,direction:'clockwise',
    }],{
      ...mklay({}),
      showlegend:false,
      annotations:[{
        text:`<b>${centerText}</b><br><span style="font-size:10px;color:${C.fg3}">top-1${expT1!==null?`<br>exp: ${expT1.toFixed(0)}%`:''}</span>`,
        showarrow:false,x:0.5,y:0.5,font:{family:'Outfit',size:18,color:C.fg},
      }],
    });
  }

  // --- Think Time vs Move Quality ---
  // Classify moves into: excellent (top-3 AND <=10cp loss), good (top-3 AND <=40cp),
  // inaccuracy (40-100cp loss), mistake/blunder (>100cp loss)
  for(const color of['white','black']){
    const tgt=$(`#time-quality-${color}`);if(!tgt) continue;
    const pr=recs.filter(r=>r.color===color&&r.scoreable===1&&isNum(r.think_time)&&r.think_time>0);
    if(pr.length<3){tgt.innerHTML='<p class="empty">No clock data available.</p>';continue}
    const cp=r=>r.delta_cp!=null?r.delta_cp:0;
    const cats=[
      {name:'Excellent',filter:r=>r.matched_t3===1&&cp(r)<=10,color:C.emerald,opacity:0.8},
      {name:'Good',filter:r=>r.matched_t3===1&&cp(r)>10&&cp(r)<=40,color:C.cyan,opacity:0.7},
      {name:'Inaccuracy',filter:r=>cp(r)>40&&cp(r)<=100,color:C.gold,opacity:0.6},
      {name:'Mistake',filter:r=>cp(r)>100,color:C.red,opacity:0.6},
    ];
    const traces=[];
    for(const cat of cats){
      const ms=pr.filter(cat.filter);
      if(ms.length) traces.push({
        x:ms.map(r=>r.think_time),y:ms.map(r=>cp(r)),
        text:ms.map(r=>r.san||''),
        mode:'markers',name:cat.name,
        marker:{size:7,color:cat.color,opacity:cat.opacity},
        hovertemplate:'%{text}: %{x:.0f}s, %{y:.0f}cp loss<extra></extra>'
      });
    }
    pl(tgt,traces,mklay({xaxis:{title:'Think time (s)'},yaxis:{title:'Centipawn loss'},legend:{orientation:'h',y:-0.18,font:{...PF,size:10}}}));
  }

  // --- Cumulative Evidence ---
  for(const color of['white','black']){
    const tgt=$(`#cumev-${color}`);if(!tgt) continue;
    const p=pl_[color];
    const pr=recs.filter(r=>r.color===color&&r.scoreable===1);
    if(!pr.length){tgt.innerHTML='<p class="empty">No data.</p>';continue}
    // Approximate per-move evidence: +1 for engine match, -1 for miss, scaled by complexity
    // Simple proxy: cumulative (matched_t1 - expected_rate) as z-like units
    const expRate=(p.summary||{}).t1||0.5;
    const baserate=p.t1_test?(p.t1_test.expected/pr.length):expRate;
    let cum=0;
    const xs=[], ys=[], sans=[];
    for(const r of pr){
      cum+=(r.matched_t1-baserate);
      xs.push(r.move_number);
      ys.push(cum);
      sans.push(r.san);
    }
    const fillColor=cum>0?'rgba(212,80,96,0.12)':'rgba(52,211,153,0.12)';
    const lineColor=cum>0?C.red:C.emerald;
    pl(tgt,[
      {x:xs,y:ys,text:sans,mode:'lines',fill:'tozeroy',line:{color:lineColor,width:2},fillcolor:fillColor,hovertemplate:'move %{x} %{text}: %{y:+.2f}<extra></extra>'},
      {x:[xs[0],xs[xs.length-1]],y:[0,0],mode:'lines',line:{color:'rgba(255,255,255,0.15)',width:1,dash:'dash'},showlegend:false,hoverinfo:'skip'}
    ],mklay({xaxis:{title:'move'},yaxis:{title:'cumulative excess matches'},showlegend:false}));
  }

  // --- Accuracy by Game Phase ---
  for(const color of['white','black']){
    const tgt=$(`#phase-acc-${color}`);if(!tgt) continue;
    const pr=recs.filter(r=>r.color===color&&r.scoreable===1);
    if(!pr.length){tgt.innerHTML='<p class="empty">No data.</p>';continue}
    const phases=['opening','middlegame','endgame'];
    const phaseLabels=['Opening','Middlegame','Endgame'];
    const labels=[],rates=[],counts=[],barColors=[];
    for(let i=0;i<phases.length;i++){
      const pn=phases[i];
      const inBin=pr.filter(r=>(r.phase_name||'middlegame')===pn);
      labels.push(phaseLabels[i]);
      counts.push(inBin.length);
      if(inBin.length){
        const rate=inBin.filter(r=>r.matched_t1===1).length/inBin.length*100;
        rates.push(rate);
        barColors.push(rate>65?C.red:rate>45?C.gold:C.emerald);
      }else{rates.push(0);barColors.push(C.fg2)}
    }
    pl(tgt,[{
      x:labels,y:rates,type:'bar',
      marker:{color:barColors},
      text:counts.map(c=>`${c} moves`),textposition:'none',
      hovertemplate:'%{x}: %{y:.0f}% engine match (%{text})<extra></extra>'
    }],mklay({yaxis:{title:'Engine match %',range:[0,100]},showlegend:false}));
  }


  // resize handler
  window.addEventListener('resize',()=>{
    n.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));
  });
}

// ================================================================ SCAN RESULTS
function renderScanReport() {
  const app=$('#app');
  app.classList.add('has-results');
  const n=$('#report');
  n.hidden=false;
  clearMsgs();
  const sr=state.scanReport;

  const lightColor=sr.posterior>=0.5?'var(--red)':sr.posterior>=0.05?'var(--gold)':'var(--emerald)';
  const lightLabel=sr.posterior>=0.5?'Highly unusual play':sr.posterior>=0.05?'Some unusual patterns':'Normal play';
  const lightClass=sr.posterior>=0.5?'light-red':sr.posterior>=0.05?'light-yellow':'light-green';

  // Aggregated stats
  const scored=sr.games.filter(g=>!g.insufficient);
  const avgACPL=scored.length?scored.reduce((s,g)=>s+(g.acpl||0),0)/scored.length:null;
  const avgMatch=scored.length?scored.reduce((s,g)=>s+(g.t1_rate||0),0)/scored.length:null;
  const iprGames=(sr.ipr_trajectory||[]).filter(t=>t.ipr!=null);
  const avgIPR=iprGames.length?iprGames.reduce((s,t)=>s+t.ipr,0)/iprGames.length:null;
  const acplArr=scored.filter(g=>g.acpl!=null);

  // Speed breakdown
  const speedBk=sr.speed_breakdown||[];
  const speedHtml=speedBk.length>=2?`
    <div class="card fade-in" style="margin-top:18px">
      <h3>Speed Breakdown</h3>
      <p class="sub">Evidence split by time control. Different time controls can produce very different signal profiles.</p>
      <div class="speed-grid">${speedBk.map(s=>{
        const dbCol=s.total_db>3?'var(--red)':s.total_db<-3?'var(--emerald)':'var(--fg-2)';
        return `<div class="speed-chip">
          <div class="sc-speed">${esc(s.speed)}</div>
          <div class="sc-val">${s.n_games} game${s.n_games!==1?'s':''}</div>
          <div class="sc-val" style="color:${dbCol}">${signed(s.total_db,1)} dB</div>
          ${s.mean_acpl!=null?`<div class="sc-val">${fmt(s.mean_acpl,0)} ACPL</div>`:''}
        </div>`;
      }).join('')}</div>
    </div>`:'';

  // Temporal changepoint
  const temp=sr.temporal;
  const temporalHtml=temp&&temp.detected?`
    <div class="card fade-in" style="margin-top:18px">
      <h3>Temporal Changepoint</h3>
      <p class="sub">Evidence concentration shifted during the game sequence.</p>
      <div class="chart short" id="temporal-chart"></div>
      <div class="temporal-note">${esc(temp.detail)}</div>
    </div>`:'';

  // Per-game table with Analyse button
  const flaggedSet=new Set(sr.flagged_games||[]);
  const gamesHtml=sr.games.map(g=>{
    const gi=g.game_info;
    const opponent=g.color==='white'?gi.black:gi.white;
    const dbCol=g.decibans>3?'var(--red)':g.decibans<-3?'var(--emerald)':'var(--fg-2)';
    const totalMoves=gi.moves?Math.ceil(gi.moves/2):'?';
    const status=g.insufficient?`<span style="color:var(--gold);font-size:10px" title="Fewer than 8 scoreable positions after excluding book moves and forced replies">too few positions</span>`:'';
    const analyseBtn=g.raw_pgn?`<button class="btn-analyse" data-gi="${g.game_index}">Analyse</button>`:'';
    const isFlagged=flaggedSet.has(g.game_index);
    const rowStyle=isFlagged?'background:rgba(255,60,60,0.12);border-left:3px solid var(--red)':'';
    const flagIcon=isFlagged?'<span title="Individually suspicious" style="color:var(--red);font-weight:700">&#9888; </span>':'';
    return `<tr style="${rowStyle}">
      <td>${flagIcon}${g.game_index+1}</td>
      <td>${esc(opponent||'?')}</td>
      <td>${g.color}</td>
      <td class="num">${totalMoves}</td>
      <td class="num">${g.n_scoreable||'--'} ${status}</td>
      <td class="num">${g.t1_rate!==null?pct(g.t1_rate):'--'}</td>
      <td class="num">${g.acpl!==null?fmt(g.acpl,0):'--'}</td>
      <td class="num" style="color:${dbCol}">${g.insufficient?'--':signed(g.decibans,1)}</td>
      <td class="num">${pct(g.posterior_cumulative)}</td>
      <td>${analyseBtn}</td>
    </tr>`;
  }).join('');

  const convDesc=sr.posterior<0.05
    ?'Posterior probability stayed near the prior throughout \u2014 consistent with normal play.'
    :'How the posterior probability changes as more games are analysed.';

  n.innerHTML=`
    <div id="gamehead">
      <div class="names">${esc(sr.username)}</div>
      <div class="meta">Player Scan &mdash; ${sr.n_games} games analysed (${sr.n_games_scored} scored) &mdash; ${sr.total_positions} positions</div>
    </div>

    <!-- VERDICT -->
    <div class="card verdict-card fade-in" style="max-width:500px;margin:0 auto 24px">
      <div class="traffic-light ${lightClass}">
        <div class="light-dot" style="background:${lightColor};box-shadow:0 0 24px ${lightColor}"></div>
        <div class="light-label">${lightLabel}</div>
      </div>
      <p class="verdict-interp" style="font-size:14px;margin-top:12px">${esc(sr.interpretation)}</p>
      <div class="verdict-detail" style="margin-top:12px">
        <span>Posterior: <strong>${pct(sr.posterior)}</strong></span>
        <span>CI: <strong>${pct(sr.posterior_low)} &ndash; ${pct(sr.posterior_high)}</strong></span>
        <span>Evidence: <strong>${signed(sr.total_decibans,1)} db</strong></span>
      </div>
    </div>

    ${(sr.flagged_games&&sr.flagged_games.length)?`
    <!-- FLAGGED GAMES WARNING -->
    <div class="card fade-in" style="max-width:600px;margin:0 auto 18px;border:1px solid var(--red);background:rgba(255,60,60,0.08)">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <span style="font-size:22px;color:var(--red)">&#9888;</span>
        <h3 style="margin:0;color:var(--red)">Per-Game Flags</h3>
      </div>
      <p style="font-size:13px;margin:0 0 8px">
        ${sr.flagged_games.length} game(s) individually exceed 50% posterior probability,
        meaning each would be suspicious even without any other games.
        The aggregate verdict may be clean because ${sr.n_games_scored - sr.flagged_games.length}
        other games are unremarkable, but these specific games warrant individual review.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        ${sr.flagged_games.map(gi=>{
          const g=sr.games.find(x=>x.game_index===gi);
          const opp=g?(g.color==='white'?g.game_info.black:g.game_info.white):'?';
          const db=g?g.decibans:0;
          return `<span style="background:rgba(255,60,60,0.18);padding:4px 10px;border-radius:6px;font-size:12px;color:var(--red)">
            Game ${gi+1} vs ${esc(opp||'?')} &mdash; ${signed(db,1)} dB
          </span>`;
        }).join('')}
      </div>
    </div>
    `:''}

    <!-- PLAYER PROFILE -->
    <div class="card fade-in" style="max-width:800px;margin:0 auto 18px">
      <h3>Player Profile</h3>
      <p class="sub">Aggregated statistics across ${sr.n_games_scored} scored game${sr.n_games_scored!==1?'s':''}</p>
      <div class="profile-stats">
        <div class="profile-stat">
          <span class="ps-val">${avgACPL!==null?fmt(avgACPL,0):'--'}</span>
          <span class="ps-label">Avg ACPL</span>
          <span class="ps-desc">How many points lost per move compared to the best move. Lower means more accurate play. Under 25 is strong; over 50 is loose.</span>
        </div>
        <div class="profile-stat">
          <span class="ps-val">${avgMatch!==null?pct(avgMatch):'--'}</span>
          <span class="ps-label">Engine Match</span>
          <span class="ps-desc">How often the player picked the same move the engine would. Most humans score 40\u201365% depending on skill level.</span>
        </div>
        <div class="profile-stat">
          <span class="ps-val">${avgIPR!==null?Math.round(avgIPR):'--'}</span>
          <span class="ps-label">Avg IPR</span>
          <span class="ps-desc">Estimated rating based purely on move quality. If this is way higher than the player's actual rating, something may be off.</span>
        </div>
        <div class="profile-stat">
          <span class="ps-val">${sr.n_games_scored}/${sr.n_games}</span>
          <span class="ps-label">Games Scored</span>
          <span class="ps-desc">How many games had enough moves to analyse. Very short games or quick resignations are skipped.</span>
        </div>
      </div>
    </div>

    <!-- ACPL PER GAME -->
    ${acplArr.length>=2?`
    <div class="card fade-in" style="margin-top:18px">
      <h3>Accuracy per Game</h3>
      <p class="sub">Average centipawn loss for each game. Lower is more accurate play.</p>
      <div class="chart short" id="acpl-chart"></div>
    </div>`:''}

    <!-- IPR TRAJECTORY -->
    ${iprGames.length>=2?`
    <div class="card fade-in" style="margin-top:18px">
      <h3>Intrinsic Performance Rating</h3>
      <p class="sub">IPR estimates playing strength from move quality, independent of opponent or result.</p>
      <div class="chart short" id="ipr-chart"></div>
    </div>`:''}

    <!-- CONVERGENCE CHART -->
    <div class="card fade-in" style="margin-top:18px">
      <h3>Evidence Convergence</h3>
      <p class="sub">${convDesc}</p>
      <div class="chart" id="convergence-chart"></div>
    </div>

    <!-- SPEED BREAKDOWN -->
    ${speedHtml}

    <!-- TEMPORAL CHANGEPOINT -->
    ${temporalHtml}

    <!-- SIGNAL SUMMARY -->
    ${sr.signal_summary.length?`
    <div class="card fade-in" style="margin-top:18px">
      <h3>Signal Summary</h3>
      <p class="sub">Aggregated z-scores across all games. Higher combined |z| = stronger signal.</p>
      <div id="signal-bars"></div>
    </div>`:''}

    <!-- PER-GAME TABLE -->
    <div class="card fade-in" style="margin-top:18px">
      <h3>Per-Game Breakdown</h3>
      <p class="sub">Evidence from each individual game. Click Analyse to drill into any game.</p>
      <div class="scroll-x">
        <table class="tbl">
          <thead><tr><th>#</th><th>Opponent</th><th>Color</th><th class="num">Moves</th><th class="num">Scored</th><th class="num">Engine Match</th><th class="num">ACPL</th><th class="num">db</th><th class="num">Cumul. Post.</th><th></th></tr></thead>
          <tbody>${gamesHtml}</tbody>
        </table>
      </div>
    </div>

    <!-- CAVEATS -->
    ${sr.caveats.length?sr.caveats.map(c=>`<div class="caveat">${esc(c)}</div>`).join(''):''}

    <!-- EXPORT -->
    <div class="export-bar">
      <a class="btn-ghost" href="/api/jobs/${state.jobId}/report.json" download>Download JSON Report</a>
    </div>
  `;

  // --- Wire drill-down Analyse buttons ---
  for(const btn of $$('.btn-analyse')){
    btn.addEventListener('click',()=>{
      const gi=parseInt(btn.dataset.gi,10);
      const game=sr.games.find(g=>g.game_index===gi);
      if(!game||!game.raw_pgn)return;
      setMode('paste');setAnalysisType('single');
      $('#pgn').value=game.raw_pgn;
      inspectPasted().then(()=>{
        $('#input-overlay').classList.remove('hidden');
        if($('#app').classList.contains('has-results'))$('#input-overlay').classList.add('visible');
      });
    });
  }

  // --- ACPL per game chart ---
  if(acplArr.length>=2){
    const el=$('#acpl-chart');
    if(el){
      const labels=acplArr.map(g=>`G${g.game_index+1}`);
      const vals=acplArr.map(g=>g.acpl||0);
      const colors=vals.map(v=>v>40?C.red:v>25?C.gold:C.cyan);
      pl(el,[{
        x:labels,y:vals,type:'bar',
        marker:{color:colors,line:{width:0}},
        hovertemplate:'%{x}: %{y:.0f} ACPL<extra></extra>',
      }],mklay({xaxis:{title:''},yaxis:{title:'ACPL (centipawns)',rangemode:'tozero'},showlegend:false}));
    }
  }

  // --- IPR trajectory chart with smoothed trend ---
  if(iprGames.length>=2){
    const el=$('#ipr-chart');
    if(el){
      const xs=iprGames.map(t=>t.game), ys=iprGames.map(t=>t.ipr);
      const traces=[{
        x:xs,y:ys,
        mode:'lines+markers',line:{color:C.cyan,width:2.5},marker:{size:7,color:C.cyan},
        name:'IPR (move quality)',hovertemplate:'Game %{x}: IPR %{y:.0f}<extra></extra>',
      }];
      // Rolling average (window=5 or fewer)
      if(ys.length>=4){
        const win=Math.min(5,Math.floor(ys.length/2));
        const smooth=[];
        for(let i=0;i<ys.length;i++){
          const lo=Math.max(0,i-Math.floor(win/2)), hi=Math.min(ys.length,i+Math.ceil(win/2));
          smooth.push(ys.slice(lo,hi).reduce((a,b)=>a+b,0)/(hi-lo));
        }
        traces.push({
          x:xs,y:smooth,
          mode:'lines',line:{color:C.teal,width:2,dash:'dash'},
          name:'Trend (rolling avg)',hovertemplate:'Game %{x}: Trend %{y:.0f}<extra></extra>',
        });
      }
      const ratedPts=iprGames.filter(t=>t.rated);
      if(ratedPts.length>=2){
        traces.push({
          x:ratedPts.map(t=>t.game),y:ratedPts.map(t=>parseInt(t.rated)),
          mode:'lines+markers',line:{color:'rgba(255,255,255,0.3)',width:1.5,dash:'dot'},
          marker:{size:5,color:'rgba(255,255,255,0.3)'},
          name:'Actual Rating',hovertemplate:'Game %{x}: Rated %{y:.0f}<extra></extra>',
        });
      }
      pl(el,traces,mklay({
        xaxis:{title:'game',dtick:iprGames.length>30?5:1},yaxis:{title:'Elo'},showlegend:true,
        legend:{x:0,y:1.15,orientation:'h',font:{color:'#6b8095',size:10}},
      }));
    }
  }

  // --- Convergence chart ---
  const conv=sr.convergence.filter(c=>!c.insufficient);
  if(conv.length){
    pl($('#convergence-chart'),[
      {x:conv.map(c=>c.game),y:conv.map(c=>c.posterior*100),text:conv.map(c=>c.label),mode:'lines+markers',line:{color:C.cyan,width:2.5},marker:{size:7,color:C.cyan},fill:'tozeroy',fillcolor:'rgba(34,211,238,0.08)',hovertemplate:'Game %{x}: %{y:.1f}%<br>%{text}<extra></extra>'},
      {x:[conv[0].game,conv[conv.length-1].game],y:[sr.prior*100,sr.prior*100],mode:'lines',line:{color:'rgba(255,255,255,0.2)',width:1,dash:'dash'},name:'Prior',hoverinfo:'skip'},
    ],mklay({xaxis:{title:'games analysed',dtick:conv.length>30?5:1},yaxis:{title:'posterior probability (%)',range:[0,Math.max(conv.reduce((m,c)=>Math.max(m,c.posterior*100),0)*1.3,5)]},showlegend:false}));
  }else{
    const ce=$('#convergence-chart');
    if(ce) ce.innerHTML='<p class="empty" style="text-align:center;padding:40px 0">Not enough scored games to plot convergence.</p>';
  }

  // --- Temporal changepoint chart ---
  if(temp&&temp.detected&&scored.length>=6){
    const tel=$('#temporal-chart');
    if(tel){
      const xs=scored.map(g=>g.game_index+1), ys=scored.map(g=>g.decibans);
      const colors=ys.map(v=>v>3?C.red:v<-3?C.emerald:C.fg2);
      const traces=[{
        x:xs,y:ys,type:'bar',marker:{color:colors},
        hovertemplate:'Game %{x}: %{y:+.1f} dB<extra></extra>',
      }];
      if(temp.split_game!=null){
        traces.push({
          x:[temp.split_game+0.5,temp.split_game+0.5],y:[Math.min(...ys)-2,Math.max(...ys)+2],
          mode:'lines',line:{color:C.gold,width:2,dash:'dash'},name:'Changepoint',
          hoverinfo:'skip',
        });
      }
      pl(tel,traces,mklay({xaxis:{title:'game'},yaxis:{title:'dB per game'},showlegend:false}));
    }
  }

  // --- Signal bars ---
  if(sr.signal_summary.length){
    const sb=$('#signal-bars');
    const maxZ=Math.max(1,...sr.signal_summary.map(s=>Math.abs(s.combined_z)));
    sb.innerHTML=sr.signal_summary.map(s=>{
      const w=clamp(Math.abs(s.combined_z)/maxZ*100,3,100);
      const col=s.combined_z>1.5?'var(--red)':s.combined_z<-1.5?'var(--emerald)':'var(--fg-3)';
      return `<div class="sig-bar-row">
        <span class="sig-bar-name">${esc(s.name.replace(/_/g,' '))}</span>
        <div class="sig-bar-track"><div class="sig-bar-fill" style="width:${w}%;background:${col}"></div></div>
        <span class="sig-bar-val" style="color:${col}">${signed(s.combined_z,2)}z (${s.n_games} games)</span>
      </div>`;
    }).join('');
  }

  window.addEventListener('resize',()=>{
    n.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));
  });
}

// ================================================================ CASE STUDIES
const caseBoards = {};
let _caseData = null;

async function loadCaseStudies() {
  try {
    const resp = await api('/api/case-studies');
    if (!resp.cases || !resp.cases.length) return;
    _caseData = resp.cases;
    renderCaseStudies(resp.cases);
    const n = $('#cases-note'); if (n) n.hidden = false;
  } catch { /* case studies are optional */ }
}

function renderCaseStudies(cases) {
  const container = $('#case-studies');
  if (!container) return;

  let flaggedIdx = 0;
  container.innerHTML = cases.map((c) => {
    const isClean = c.type === 'clean';
    const label = isClean ? 'COUNTER-EXAMPLE' : `CASE 0${++flaggedIdx}`;
    const hi = c.highlight_index || 0;
    const game = c.games[hi];
    if (!game) return '';

    const statsHtml = (c.stats || []).map(s => `
      <div class="case-stat${s.highlight ? ' highlight' : ''}">
        <span class="cs-val">${esc(s.value)}</span>
        <span class="cs-label">${esc(s.label)}</span>
      </div>
    `).join('');

    const opts = c.games.map((g, i) =>
      `<option value="${i}" ${i===hi?'selected':''}>${esc(g.white)} vs ${esc(g.black)}${g.round ? ' (R'+esc(g.round)+')' : ''} ${g.result}</option>`
    ).join('');

    return `
      <div class="case-group${isClean ? ' case-group--clean' : ''} fade-in">
        <div class="case-card" data-case="${esc(c.id)}">
          <div class="case-info-col">
            <div class="case-group-header">
              <span class="case-group-num">${label}</span>
              <h3 class="case-group-title">${esc(c.title)}</h3>
            </div>
            <div class="case-group-sub">${esc(c.subtitle)}</div>
            <p class="case-group-narrative">${esc(c.narrative)}</p>
            <div class="case-stats">${statsHtml}</div>
            <div class="case-detail"><strong>Method:</strong> ${esc(c.method || '')}</div>
            <div class="case-detail"><strong>Outcome:</strong> ${esc(c.outcome || '')}</div>
          </div>
          <div class="case-board-col">
            <div class="board-player board-player--top" id="case-player-top-${c.id}">
              <span class="board-player-name">${esc(game.black)}</span>
              <span class="board-player-elo">${esc(game.black_elo)}</span>
            </div>
            <div class="case-board-wrap" id="case-board-${c.id}"></div>
            <div class="board-player board-player--bottom" id="case-player-bottom-${c.id}">
              <span class="board-player-name">${esc(game.white)}</span>
              <span class="board-player-elo">${esc(game.white_elo)}</span>
            </div>
            <div class="case-controls">
              <button data-action="rwd" data-case="${c.id}">&#9198;</button>
              <button data-action="prev" data-case="${c.id}">&#9664;</button>
              <button data-action="next" data-case="${c.id}">&#9654;</button>
              <button data-action="ffwd" data-case="${c.id}">&#9197;</button>
            </div>
            <div class="case-ply" id="case-ply-${c.id}"></div>
            ${c.games.length > 1 ? `<select class="case-game-select" data-case="${esc(c.id)}">${opts}</select>` : ''}
            <button class="btn-ghost case-load-btn" data-case="${c.id}" style="width:100%;justify-content:center;margin-top:6px">Load in Analyser</button>
          </div>
        </div>
      </div>`;
  }).join('');

  // Init boards
  loadBoardLib().then(lib => {
    for (const c of cases) initCaseBoard(c, lib);
  }).catch(() => {});

  // Board controls
  for (const btn of $$('.case-controls button')) {
    btn.addEventListener('click', () => {
      const id = btn.dataset.case, cb = caseBoards[id];
      if (!cb) return;
      const act = btn.dataset.action;
      if (act === 'rwd') setCasePly(id, 0);
      else if (act === 'prev') setCasePly(id, cb.ply - 1);
      else if (act === 'next') setCasePly(id, cb.ply + 1);
      else if (act === 'ffwd') setCasePly(id, cb.moves.length - 1);
    });
  }

  // Game selectors
  for (const sel of $$('.case-game-select')) {
    sel.addEventListener('change', () => {
      const cid = sel.dataset.case;
      const caseObj = cases.find(c => c.id === cid);
      if (caseObj) switchCaseGame(cid, caseObj, parseInt(sel.value, 10));
    });
  }

  // Load buttons
  for (const btn of $$('.case-load-btn')) {
    btn.addEventListener('click', () => {
      const caseObj = cases.find(c => c.id === btn.dataset.case);
      if (!caseObj || !caseObj.raw_pgn) return;
      const sel = $(`.case-game-select[data-case="${btn.dataset.case}"]`);
      const gi = sel ? parseInt(sel.value, 10) : 0;
      loadCasePgn(caseObj, gi);
    });
  }
}

function initCaseBoard(caseObj, lib) {
  const id = caseObj.id;
  const gi = caseObj.highlight_index || 0;
  const game = caseObj.games[gi];
  if (!game || !game.moves.length) return;
  const el = $(`#case-board-${id}`);
  if (!el) return;
  const board = new lib.Chessboard(el, {
    position: game.moves[0].fen_before,
    assetsUrl: 'https://cdn.jsdelivr.net/npm/cm-chessboard@8.7.5/assets/',
    style: { cssClass: 'default', borderType: lib.BORDER_TYPE.none, pieces: { file: 'pieces/staunty.svg' } },
    extensions: [{ class: lib.Markers }],
  });
  caseBoards[id] = { board, lib, ply: 0, moves: game.moves, finalFen: game.final_fen };
  setCasePly(id, 0);
}

function switchCaseGame(caseId, caseObj, gi) {
  const game = caseObj.games[gi];
  if (!game) return;
  const cb = caseBoards[caseId];
  if (!cb) return;
  cb.moves = game.moves;
  cb.finalFen = game.final_fen;
  cb.ply = 0;
  const top = $(`#case-player-top-${caseId}`);
  const bot = $(`#case-player-bottom-${caseId}`);
  if (top) top.innerHTML = `<span class="board-player-name">${esc(game.black)}</span><span class="board-player-elo">${esc(game.black_elo)}</span>`;
  if (bot) bot.innerHTML = `<span class="board-player-name">${esc(game.white)}</span><span class="board-player-elo">${esc(game.white_elo)}</span>`;
  if (game.moves.length) {
    cb.board.setPosition(game.moves[0].fen_before, false);
    cb.board.removeMarkers();
  }
  setCasePly(caseId, 0);
}

function setCasePly(caseId, ply) {
  const cb = caseBoards[caseId];
  if (!cb || !cb.moves.length) return;
  ply = clamp(Math.round(ply), 0, cb.moves.length - 1);
  cb.ply = ply;
  const m = cb.moves[ply];
  const fenAfter = cb.moves[ply + 1] ? cb.moves[ply + 1].fen_before : cb.finalFen;
  cb.board.setPosition(fenAfter || m.fen_before, false);
  cb.board.removeMarkers();
  if (m.uci && m.uci.length >= 4) {
    cb.board.addMarker(cb.lib.MARKER_TYPE.frame, m.uci.slice(0, 2));
    cb.board.addMarker(cb.lib.MARKER_TYPE.frame, m.uci.slice(2, 4));
  }
  const info = $(`#case-ply-${caseId}`);
  if (info) {
    const mn = Math.floor(ply / 2) + 1;
    info.textContent = `${mn}${ply % 2 ? '...' : '.'} ${m.san}  (${ply + 1}/${cb.moves.length})`;
  }
}

async function loadCasePgn(caseObj, gi) {
  setMode('paste');
  setAnalysisType('single');
  $('#pgn').value = caseObj.raw_pgn;
  await inspectPasted();
  const pick = $('#pick');
  if (pick && gi >= 0 && gi < (pick.options||[]).length) pick.value = String(gi);
  $('#input-overlay').classList.remove('hidden');
}

// ================================================================ BOOTSTRAP
async function loadBaseline() {
  try {
    const b = await api('/api/baseline');
    const flag = b.is_default ? '<span class="badge suspect">uncalibrated</span>'
      : b.provisional ? '<span class="badge neutral">provisional</span>'
      : '<span class="badge ok">fitted</span>';
    $('#baseinfo').innerHTML = `${flag} ${esc(b.description)}`;
    if (isNum(b.default_prior)) $('#prior').value = b.default_prior;
    // Update methodology corpus stats
    if (b.games) {
      const el = document.getElementById('corpus-games');
      if (el) el.textContent = b.games >= 1e6
        ? (b.games / 1e6).toFixed(1) + 'M' : b.games.toLocaleString();
    }
    if (b.positions) {
      const el = document.getElementById('corpus-positions');
      if (el) el.textContent = b.positions >= 1e6
        ? (b.positions / 1e6).toFixed(1) + 'M' : b.positions.toLocaleString();
    }
    if (b.bands_fitted) {
      const el = document.getElementById('corpus-bands');
      if (el) el.textContent = b.bands_fitted;
    }
  } catch (e) { $('#baseinfo').textContent = `Error: ${e.message}`; }
}

function wire() {
  // Analysis type toggle (single / scan)
  $$('#analysis-type button').forEach(b => b.addEventListener('click', () => setAnalysisType(b.dataset.type)));
  // Scan source toggle (chesscom / lichess)
  $$('#scan-source button').forEach(b => b.addEventListener('click', () => setScanSource(b.dataset.src)));

  $$('#modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#pgn').addEventListener('change', inspectPasted);
  $('#pgnfile').addEventListener('change', async ev => {
    const f = ev.target.files[0]; if (!f) return;
    $('#pgn').value = await f.text(); await inspectPasted();
  });
  $('#fetchurl').addEventListener('click', () => doImport('url', $('#gameurl').value.trim(), 1));
  $('#fetchcc').addEventListener('click', () => doImport('chesscom', $('#ccuser').value.trim(), parseInt($('#cccount').value,10)||10, $('#ccspeed').value));
  $('#fetchlc').addEventListener('click', () => doImport('lichess', $('#lcuser').value.trim(), parseInt($('#lccount').value,10)||10, $('#lcspeed').value));
  $('#run').addEventListener('click', runAnalysis);

  // Cancel button
  const cancelBtn=$('#cancel-scan');
  if(cancelBtn) cancelBtn.addEventListener('click',async()=>{
    if(state.jobId){
      try{await api(`/api/jobs/${state.jobId}/cancel`,{})}catch{}
    }
    cancelBtn.hidden=true;
  });

  function showOverlay() {
    const ov = $('#input-overlay');
    ov.classList.remove('hidden');
    if ($('#app').classList.contains('has-results')) ov.classList.add('visible');
  }
  function hideOverlay() {
    const ov = $('#input-overlay');
    ov.classList.add('hidden');
    ov.classList.remove('visible');
  }
  $('#open-input').addEventListener('click', showOverlay);
  $('#hero-cta').addEventListener('click', showOverlay);
  const closeBtn = $('#close-overlay');
  if (closeBtn) closeBtn.addEventListener('click', hideOverlay);
  $('#input-overlay').addEventListener('click', e => {
    if (e.target === $('#input-overlay')) hideOverlay();
  });

  // Nav links: if results are showing, go back to landing first
  $$('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      const app = $('#app');
      if (app.classList.contains('has-results')) {
        e.preventDefault();
        app.classList.remove('has-results');
        $('#report').hidden = true;
        // Scroll to target after landing is visible again
        const target = link.getAttribute('href');
        requestAnimationFrame(() => {
          const el = document.querySelector(target);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        });
      }
    });
  });
}

wire();
setScanSource('chesscom'); // populate initial scan speed options
loadBaseline();
loadCaseStudies();

// Scroll-reveal observer for landing page sections
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); revealObs.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll('.features-section h2.sec, .features-section p.secsub, .features-grid, .cases-section h2.sec, .cases-section p.secsub, .case-grid, .method-block, .landing-footer').forEach(el => {
  el.classList.add('reveal');
  revealObs.observe(el);
});
// Also trigger waterfall animation when method-section scrolls in
const methObs = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); methObs.unobserve(e.target); } });
}, { threshold: 0.1 });
const ms = document.querySelector('.method-section');
if (ms) methObs.observe(ms);
// "Try it now" button in methodology
const methodCta = document.getElementById('method-cta');
if (methodCta) methodCta.addEventListener('click', () => {
  const ov = document.getElementById('input-overlay');
  if (ov) { ov.classList.remove('hidden'); ov.classList.add('visible'); }
});
