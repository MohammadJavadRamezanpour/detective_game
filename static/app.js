let gameId = null;
let suspects = [];
let suspicion = {};

const el = (sel) => document.querySelector(sel);

const $list = el('#suspectList');
const $summary = el('#summaryText');
const $messages = el('#messages');
const $suspectSelect = el('#suspectSelect');
const $questionInput = el('#questionInput');
const $status = el('#status');
const $askBtn = el('#askBtn');
const $accuseBtn = el('#accuseBtn');

/* =========================
   RENDERING
========================= */

function renderSuspects() {
  $suspectSelect.innerHTML = '';
  $list.innerHTML = '';

  suspects.forEach(s => {
    // dropdown option
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `${s.name} (${s.occupation})`;
    $suspectSelect.appendChild(opt);

    // suspect card
    const card = document.createElement('div');
    card.className = 'suspect';
    card.innerHTML = `
      <div class="name">🧍 ${s.name}</div>
      <div class="role">${s.occupation}</div>
      <div class="meter">
        <div class="fill" id="meter-${s.id}"></div>
      </div>
    `;
    $list.appendChild(card);
  });

  updateMeters();
}

function updateMeters() {
  Object.entries(suspicion).forEach(([id, val]) => {
    const pct = Math.min(100, Math.max(0, (val / 10) * 100));
    const fill = el(`#meter-${id}`);

    if (fill) {
      fill.style.width = `${pct}%`;

      // pulse animation
      fill.animate(
        [
          { boxShadow: '0 0 0px transparent' },
          { boxShadow: '0 0 12px rgba(255,77,77,0.9)' },
          { boxShadow: '0 0 0px transparent' }
        ],
        { duration: 350 }
      );
    }
  });
}

/* =========================
   CHAT
========================= */

function appendMsg(who, content) {
  const div = document.createElement('div');
  div.className = 'msg';
  div.innerHTML = `
    <span class="who">${who}:</span>
    <span class="content">${content}</span>
  `;
  $messages.appendChild(div);
  $messages.scrollTop = $messages.scrollHeight;
}

function setStatus(text, emoji = '') {
  $status.textContent = `${emoji ? emoji + ' ' : ''}${text}`;
}

/* =========================
   GAME ACTIONS
========================= */

async function newGame() {
  setStatus('Creating a new case…', '🗂️');
  $messages.innerHTML = '';
  $summary.textContent = '';
  enableInputs();

  try {
    const res = await fetch('/api/new_game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_suspects: 4 })
    });

    const data = await res.json();

    gameId = data.game_id;
    suspects = data.suspects;
    suspicion = data.suspicion || {};

    $summary.textContent = data.summary || '';
    renderSuspects();

    setStatus('Case ready. Start questioning.', '🕵️');
    $questionInput.focus();
  } catch (err) {
    console.error(err);
    setStatus('Failed to start game.', '❌');
  }
}

async function ask() {
  if (!gameId) return;

  const suspectId = $suspectSelect.value;
  const question = $questionInput.value.trim();
  if (!question) return;

  $questionInput.value = '';
  appendMsg('You', question);
  setStatus('Waiting for reply…', '⌛');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_id: gameId,
        suspect_id: suspectId,
        question
      })
    });

    const data = await res.json();
    const suspect = suspects.find(s => s.id === suspectId);

    appendMsg(suspect ? suspect.name : 'Suspect', data.answer);

    suspicion = data.suspicion || suspicion;
    updateMeters();

    if (data.game_over) {
      setStatus(
        data.result === 'win' ? 'You cracked the case!' : 'The culprit escaped…',
        data.result === 'win' ? '🏆' : '💀'
      );
      disableInputs();
    } else {
      setStatus('Reply received.', '💬');
    }
  } catch (err) {
    console.error(err);
    setStatus('Question failed.', '❌');
  }
}

async function accuse() {
  if (!gameId) return;

  // dramatic screen shake
  document.body.classList.add('shake');
  setTimeout(() => document.body.classList.remove('shake'), 300);

  setStatus('Making accusation…', '⚖️');

  try {
    const res = await fetch('/api/accuse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_id: gameId,
        suspect_id: $suspectSelect.value
      })
    });

    const data = await res.json();

    if (data.game_over) {
      appendMsg(
        'Case',
        data.messages?.slice(-1)[0]?.content || 'The case is closed.'
      );

      setStatus(
        data.result === 'win' ? 'Justice served!' : 'Wrong call…',
        data.result === 'win' ? '🎉' : '☠️'
      );

      disableInputs();
    } else {
      setStatus('Accusation rejected.', '❗');
    }
  } catch (err) {
    console.error(err);
    setStatus('Accusation failed.', '❌');
  }
}

/* =========================
   INPUT STATE
========================= */

function disableInputs() {
  $askBtn.disabled = true;
  $accuseBtn.disabled = true;
  $questionInput.disabled = true;
  $suspectSelect.disabled = true;
}

function enableInputs() {
  $askBtn.disabled = false;
  $accuseBtn.disabled = false;
  $questionInput.disabled = false;
  $suspectSelect.disabled = false;
}

/* =========================
   EVENTS
========================= */

document.addEventListener('DOMContentLoaded', () => {
  el('#newGameBtn').addEventListener('click', newGame);
  $askBtn.addEventListener('click', ask);
  $accuseBtn.addEventListener('click', accuse);

  // press Enter to ask
  $questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') ask();
  });
});
