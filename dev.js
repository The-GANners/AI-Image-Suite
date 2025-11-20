const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

function asciiSafe(line) {
  return line.toString().replace(/[^\x20-\x7E]/g, '?');
}

const suiteDir = __dirname;

// NEW: Try multiple common venv names (order matters - most common first)
const venvCandidates = [
  path.join(suiteDir, 'venv', 'Scripts', 'python.exe'),     // Your venv
  path.join(suiteDir, 'myenv', 'Scripts', 'python.exe'),    // Friends' venv
];

// Find first existing venv python
let pythonExe = 'python'; // Fallback to system python
for (const candidate of venvCandidates) {
  if (fs.existsSync(candidate)) {
    pythonExe = candidate;
    console.log(`[DEV] Found venv at: ${candidate}`);
    break;
  }
}

if (pythonExe === 'python') {
  console.warn('[DEV] No venv found! Trying system python (may cause dependency issues)');
  console.warn('[DEV] Searched paths:', venvCandidates);
}

const backendScript = path.join(suiteDir, 'server', 'app.py');

// Start backend
console.log('[DEV] Starting backend:', pythonExe, backendScript);
const backend = spawn(pythonExe, [backendScript], {
  cwd: suiteDir,
  env: {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONIOENCODING: 'utf-8'
  },
  stdio: 'inherit'
});

backend.on('error', (err) => {
  console.error('[BACKEND-ERR] spawn error:', err.message);
  console.error('[BACKEND-ERR] Ensure venv exists. Tried:', venvCandidates);
  console.error('[BACKEND-ERR] Run: python -m venv venv (or myenv)');
});
backend.on('exit', code => console.log('[BACKEND] exited code=' + code));

// Start frontend
console.log('[DEV] Starting frontend: npm start');
const frontend = spawn('npm start', {
  cwd: suiteDir,
  env: { ...process.env },
  shell: true,
  stdio: 'inherit'
});

frontend.on('error', (err) => {
  console.error('[FRONTEND-ERR] spawn error:', err.message);
});
frontend.on('exit', code => console.log('[FRONTEND] exited code=' + code));

// Graceful shutdown
function shutdown() {
  console.log('[DEV] Shutting down...');
  try { backend.kill(); } catch {}
  try { frontend.kill(); } catch {}
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
