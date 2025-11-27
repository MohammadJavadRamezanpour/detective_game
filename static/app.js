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

function renderSuspects() {
  $suspectSelect.innerHTML = '';
  $list.innerHTML = '';
  suspects.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `${s.name} (${s.occupation})`;
    $suspectSelect.appendChild(opt);

    const card = document.createElement('div');
    card.className = 'suspect';
    card.innerHTML = `
      <div class="name">${s.name}</div>
      <div class="role">${s.occupation}</div>
      <div class="meter"><div class="fill" id="meter-${s.id}"></div></div>
    `;
    $list.appendChild(card);
  });
  updateMeters();
}

function updateMeters() {
  Object.entries(suspicion).forEach(([id, val]) => {
    const pct = Math.min(100, Math.max(0, (val / 10) * 100));
    const fill = el(`#meter-${id}`);
    if (fill) fill.style.width = `${pct}%`;
  });
}

function appendMsg(who, content) {
  const div = document.createElement('div');
  div.className = 'msg';
  div.innerHTML = `<span class="who">${who}:</span> <span class="content">${content}</span>`;
  $messages.appendChild(div);
  $messages.scrollTop = $messages.scrollHeight;
}

async function newGame() {
  $status.textContent = 'Creating a new case...';
  $messages.innerHTML = '';
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
    $status.textContent = 'Case ready. Start questioning.';
  } catch (e) {
    console.error(e);
    $status.textContent = 'Failed to start game.';
  }
}

async function ask() {
  if (!gameId) return;
  const suspectId = $suspectSelect.value;
  const question = $questionInput.value.trim();
  if (!question) return;
  $questionInput.value = '';
  appendMsg('You', question);
  $status.textContent = 'Waiting for reply...';
  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, suspect_id: suspectId, question })
    });
    const data = await res.json();
    const s = suspects.find(s => s.id === suspectId);
    appendMsg(s ? s.name : 'Suspect', data.answer);
    suspicion = data.suspicion || suspicion;
    updateMeters();
    if (data.game_over) {
      $status.textContent = data.result === 'win' ? 'You won!' : 'You lost.';
      disableInputs();
    } else {
      $status.textContent = 'Reply received.';
    }
  } catch (e) {
    console.error(e);
    $status.textContent = 'Ask failed.';
  }
}

async function accuse() {
  if (!gameId) return;
  const suspectId = $suspectSelect.value;
  $status.textContent = 'Accusing...';
  try {
    const res = await fetch('/api/accuse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, suspect_id: suspectId })
    });
    const data = await res.json();
    if (data.game_over) {
      appendMsg('Case', data.messages?.slice(-1)[0]?.content || 'Case ended.');
      $status.textContent = data.result === 'win' ? 'You won!' : 'You lost.';
      disableInputs();
    } else {
      $status.textContent = 'Accusation did not end the game.';
    }
  } catch (e) {
    console.error(e);
    $status.textContent = 'Accuse failed.';
  }
}

function disableInputs() {
  el('#askBtn').disabled = true;
  el('#accuseBtn').disabled = true;
  $questionInput.disabled = true;
  $suspectSelect.disabled = true;
}

document.addEventListener('DOMContentLoaded', () => {
  el('#newGameBtn').addEventListener('click', newGame);
  el('#askBtn').addEventListener('click', ask);
  el('#accuseBtn').addEventListener('click', accuse);
});