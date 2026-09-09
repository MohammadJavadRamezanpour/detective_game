let gameId = null;
let suspects = [];
let suspicion = {};
let details = {};
let contradictions = [];
let busy = false;

const el = (selector) => document.querySelector(selector);
const $list = el('#suspectList');
const $summary = el('#summaryText');
const $messages = el('#messages');
const $suspectSelect = el('#suspectSelect');
const $questionInput = el('#questionInput');
const $status = el('#status');
const $askBtn = el('#askBtn');
const $accuseBtn = el('#accuseBtn');
const $newGameBtn = el('#newGameBtn');
const $clueList = el('#clueList');
const $contradictionList = el('#contradictionList');
const $turnCount = el('#turnCount');

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function renderSuspects() {
  $suspectSelect.replaceChildren();
  $list.replaceChildren();

  suspects.forEach((suspect) => {
    const option = node('option', '', `${suspect.name} — ${suspect.occupation}`);
    option.value = suspect.id;
    $suspectSelect.appendChild(option);

    const card = node('button', 'suspect');
    card.type = 'button';
    card.dataset.suspectId = suspect.id;
    card.appendChild(node('div', 'name', suspect.name));
    card.appendChild(node('div', 'role', suspect.occupation));

    const scoreRow = node('div', 'score-row');
    scoreRow.appendChild(node('span', '', 'Suspicion'));
    scoreRow.appendChild(node('strong', 'score', '0.0'));
    card.appendChild(scoreRow);

    const meter = node('div', 'meter');
    const fill = node('div', 'fill');
    fill.id = `meter-${suspect.id}`;
    meter.appendChild(fill);
    card.appendChild(meter);
    card.addEventListener('click', () => {
      $suspectSelect.value = suspect.id;
      syncSelectedSuspect();
      $questionInput.focus();
    });
    $list.appendChild(card);
  });

  updateMeters();
  syncSelectedSuspect();
}

function syncSelectedSuspect() {
  document.querySelectorAll('.suspect').forEach((card) => {
    card.classList.toggle('selected', card.dataset.suspectId === $suspectSelect.value);
  });
}

function updateMeters() {
  Object.entries(suspicion).forEach(([id, value]) => {
    const percent = Math.min(100, Math.max(0, (value / 10) * 100));
    const fill = el(`#meter-${id}`);
    const card = el(`[data-suspect-id="${id}"]`);
    if (!fill || !card) return;
    fill.style.width = `${percent}%`;
    card.querySelector('.score').textContent = Number(value).toFixed(1);
    fill.animate(
      [
        { boxShadow: '0 0 0 transparent' },
        { boxShadow: '0 0 14px rgba(227, 174, 87, 0.85)' },
        { boxShadow: '0 0 0 transparent' },
      ],
      { duration: 400 },
    );
  });
}

function renderCaseFacts() {
  el('#crimeText').textContent = details.crime || 'Unknown';
  el('#locationText').textContent = details.location || 'Unknown';
  el('#timeText').textContent = details.time_window || 'Unknown';
  el('#caseFacts').hidden = false;
}

function renderEvidence() {
  $clueList.replaceChildren();
  (details.clues || []).forEach((clue) => {
    const item = node('li');
    item.appendChild(node('span', 'evidence-index', String($clueList.children.length + 1).padStart(2, '0')));
    item.appendChild(node('span', '', clue));
    $clueList.appendChild(item);
  });
  $clueList.classList.toggle('empty-list', !(details.clues || []).length);

  $contradictionList.replaceChildren();
  if (!contradictions.length) {
    $contradictionList.appendChild(node('li', '', 'No contradictions logged.'));
    $contradictionList.classList.add('empty-list');
    return;
  }
  $contradictionList.classList.remove('empty-list');
  contradictions.forEach((item) => {
    const row = node('li');
    row.appendChild(node('strong', '', item.suspect_name));
    row.appendChild(node('span', '', item.text));
    $contradictionList.appendChild(row);
  });
}

function appendMessage(who, content, kind = '') {
  const message = node('div', `msg ${kind}`.trim());
  message.appendChild(node('span', 'who', `${who}:`));
  message.appendChild(node('span', 'content', content));
  $messages.appendChild(message);
  $messages.scrollTop = $messages.scrollHeight;
}

function setStatus(text, kind = '') {
  $status.textContent = text;
  $status.className = `status ${kind}`.trim();
}

function setBusy(value) {
  busy = value;
  $newGameBtn.disabled = value;
  $askBtn.disabled = value || !gameId;
  $accuseBtn.disabled = value || !gameId;
  $questionInput.disabled = value || !gameId;
  $suspectSelect.disabled = value || !gameId;
}

function endGame() {
  $askBtn.disabled = true;
  $accuseBtn.disabled = true;
  $questionInput.disabled = true;
  $suspectSelect.disabled = true;
}

async function requestJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

async function newGame() {
  setBusy(true);
  setStatus('Generating a checkpointed case…', 'working');
  $messages.replaceChildren();
  $summary.textContent = '';
  try {
    const data = await requestJson('/api/new_game', { num_suspects: 4 });
    gameId = data.game_id;
    suspects = data.suspects;
    suspicion = data.suspicion || {};
    details = data.details || {};
    contradictions = data.contradictions || [];
    $summary.textContent = data.summary || '';
    $turnCount.textContent = '0 interviews';
    renderCaseFacts();
    renderEvidence();
    renderSuspects();
    setStatus('Case ready. Select a person of interest and test their timeline.', 'success');
    $questionInput.focus();
  } catch (error) {
    gameId = null;
    setStatus(error.message || 'Failed to create the case.', 'error');
  } finally {
    setBusy(false);
  }
}

async function ask() {
  if (!gameId || busy) return;
  const suspectId = $suspectSelect.value;
  const question = $questionInput.value.trim();
  if (question.length < 2) {
    setStatus('Enter a specific question first.', 'error');
    return;
  }

  const suspect = suspects.find((item) => item.id === suspectId);
  $questionInput.value = '';
  appendMessage('You', question, 'player');
  setBusy(true);
  setStatus(`Interviewing ${suspect?.name || 'suspect'}…`, 'working');
  try {
    const data = await requestJson('/api/ask', {
      game_id: gameId,
      suspect_id: suspectId,
      question,
    });
    appendMessage(suspect?.name || 'Suspect', data.answer, 'suspect-answer');
    suspicion = data.suspicion || suspicion;
    contradictions = data.contradictions || contradictions;
    updateMeters();
    renderEvidence();
    $turnCount.textContent = `${data.turn_count} interview${data.turn_count === 1 ? '' : 's'}`;
    setStatus(data.analysis?.rationale || 'Answer recorded.', data.analysis?.contradiction_detected ? 'alert' : 'success');
  } catch (error) {
    setStatus(error.message || 'The interview failed.', 'error');
  } finally {
    setBusy(false);
  }
}

async function accuse() {
  if (!gameId || busy) return;
  const suspectId = $suspectSelect.value;
  const suspect = suspects.find((item) => item.id === suspectId);
  if (!window.confirm(`Make a final accusation against ${suspect?.name || 'this suspect'}?`)) return;

  setBusy(true);
  setStatus('Evaluating your accusation against the case file…', 'working');
  try {
    const data = await requestJson('/api/accuse', { game_id: gameId, suspect_id: suspectId });
    const verdict = data.messages?.at(-1)?.content || 'The case is closed.';
    appendMessage('Case', verdict, data.result === 'win' ? 'verdict-win' : 'verdict-lose');
    setStatus(data.result === 'win' ? 'Case solved.' : 'The evidence led elsewhere.', data.result === 'win' ? 'success' : 'error');
    setBusy(false);
    endGame();
  } catch (error) {
    setStatus(error.message || 'The accusation failed.', 'error');
    setBusy(false);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  $newGameBtn.addEventListener('click', newGame);
  $askBtn.addEventListener('click', ask);
  $accuseBtn.addEventListener('click', accuse);
  $suspectSelect.addEventListener('change', syncSelectedSuspect);
  $questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') ask();
  });
  setBusy(false);
});
