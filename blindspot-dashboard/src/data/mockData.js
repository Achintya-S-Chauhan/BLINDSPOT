export const initialSettings = {
  pollIntervalMs: 500,
  ocrIntervalSeconds: 30,
  maxOcrCharacters: 250,
  excludedApps: [
    '1Password.exe',
    'Bitwarden.exe',
    'KeePass.exe',
    'chrome.exe (Incognito)',
    'BankingPortal',
    'Signal.exe',
  ],
  inMemoryOnly: true,
  blurSensitiveFields: true,
  soundNotifications: false,
  retentionDays: 7,
};

export const mockHourlyActivity = [
  { hour: '09:00', development: 42, research: 10, communication: 5, documentation: 3, other: 0 },
  { hour: '10:00', development: 50, research: 5, communication: 4, documentation: 1, other: 0 },
  { hour: '11:00', development: 20, research: 25, communication: 12, documentation: 3, other: 0 },
  { hour: '12:00', development: 15, research: 5, communication: 10, documentation: 5, other: 25 },
  { hour: '13:00', development: 48, research: 8, communication: 2, documentation: 2, other: 0 },
  { hour: '14:00', development: 55, research: 3, communication: 2, documentation: 0, other: 0 },
  { hour: '15:00', development: 30, research: 20, communication: 8, documentation: 2, other: 0 },
  { hour: '16:00', development: 38, research: 12, communication: 5, documentation: 5, other: 0 },
  { hour: '17:00', development: 45, research: 5, communication: 8, documentation: 2, other: 0 },
  { hour: '18:00', development: 25, research: 10, communication: 15, documentation: 10, other: 0 },
];

export const mockAppUsage = [
  {
    name: 'Visual Studio Code',
    category: 'Development',
    totalSeconds: 14520, // ~4h 2m
    formattedDuration: '4h 02m',
    color: '#6366f1',
    iconName: 'Code2',
    percentage: 54,
  },
  {
    name: 'Google Chrome',
    category: 'Research',
    totalSeconds: 5880, // ~1h 38m
    formattedDuration: '1h 38m',
    color: '#3b82f6',
    iconName: 'Globe',
    percentage: 22,
  },
  {
    name: 'Slack',
    category: 'Communication',
    totalSeconds: 2760, // 46m
    formattedDuration: '46m',
    color: '#10b981',
    iconName: 'MessageSquare',
    percentage: 10,
  },
  {
    name: 'Windows Terminal',
    category: 'Development',
    totalSeconds: 2040, // 34m
    formattedDuration: '34m',
    color: '#a855f7',
    iconName: 'Terminal',
    percentage: 8,
  },
  {
    name: 'Notion',
    category: 'Documentation',
    totalSeconds: 1560, // 26m
    formattedDuration: '26m',
    color: '#f59e0b',
    iconName: 'FileText',
    percentage: 6,
  },
];

export const mockActivityHistory = [
  {
    id: 'act-001',
    application: 'Visual Studio Code',
    windowTitle: 'ui.py - BLINDSPOT - Visual Studio Code',
    category: 'Development',
    startTime: '17:42:10',
    endTime: '18:15:30',
    durationSeconds: 2000,
    durationFormatted: '33m 20s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'def _handle_chat_response(self, res: Dict[str, Any]) -> None: if self._is_destroyed: return',
  },
  {
    id: 'act-002',
    application: 'Google Chrome',
    windowTitle: 'Gemini API Error Codes & Quotas Reference — Google Docs',
    category: 'Research',
    startTime: '17:28:00',
    endTime: '17:42:10',
    durationSeconds: 850,
    durationFormatted: '14m 10s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'HTTP 429 RESOURCE_EXHAUSTED Rate limit exceeded for model gemini-3.6-flash limit: 20 requests per day',
  },
  {
    id: 'act-003',
    application: 'Windows Terminal',
    windowTitle: 'PowerShell - python -m unittest tests.test_ui_chat',
    category: 'Development',
    startTime: '17:15:20',
    endTime: '17:28:00',
    durationSeconds: 760,
    durationFormatted: '12m 40s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'Ran 19 tests in 1.489s OK [BACKEND] Active Desktop I/O: REAL WINDOWS I/O',
  },
  {
    id: 'act-004',
    application: 'Visual Studio Code',
    windowTitle: 'ai.py - BLINDSPOT - Visual Studio Code',
    category: 'Development',
    startTime: '16:30:00',
    endTime: '17:15:20',
    durationSeconds: 2720,
    durationFormatted: '45m 20s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'class QuotaExceededError(LLMProviderError): def deterministic_local_answer()',
  },
  {
    id: 'act-005',
    application: 'Slack',
    windowTitle: '#general — Engineering Standup — Slack',
    category: 'Communication',
    startTime: '16:15:00',
    endTime: '16:30:00',
    durationSeconds: 900,
    durationFormatted: '15m 00s',
    confidence: 2,
    date: 'Today',
    ocrSnippet: 'Standup update: Phase 9A Companion Chat UI and quota fallback deterministic resolution complete.',
  },
  {
    id: 'act-006',
    application: 'Notion',
    windowTitle: 'BLINDSPOT Architecture Audit & Roadmaps 2026',
    category: 'Documentation',
    startTime: '15:45:00',
    endTime: '16:15:00',
    durationSeconds: 1800,
    durationFormatted: '30m 00s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'Stage 1: Focus application Win32 API. Stage 2: Genuine fresh verification. Stage 3: Runtime backend selection.',
  },
  {
    id: 'act-007',
    application: 'Visual Studio Code',
    windowTitle: 'desktop_io.py - BLINDSPOT',
    category: 'Development',
    startTime: '14:20:00',
    endTime: '15:45:00',
    durationSeconds: 5100,
    durationFormatted: '1h 25m',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'def set_foreground_window(self, hwnd: int) -> bool: win32gui.SetForegroundWindow(hwnd)',
  },
  {
    id: 'act-008',
    application: 'Figma',
    windowTitle: 'BLINDSPOT Desktop UI System v2 - Light & Dark Tokens',
    category: 'Design',
    startTime: '13:30:00',
    endTime: '14:20:00',
    durationSeconds: 3000,
    durationFormatted: '50m 00s',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'Cards padding: 12px. Accent Blue: #7aa2f7. Dark Root: #1a1b26. Status Badge border-radius: 9999px.',
  },
  {
    id: 'act-009',
    application: 'Google Chrome',
    windowTitle: 'Stack Overflow - How to prevent Tcl_AsyncDelete in multi-threaded Tkinter',
    category: 'Research',
    startTime: '13:00:00',
    endTime: '13:30:00',
    durationSeconds: 1800,
    durationFormatted: '30m 00s',
    confidence: 2,
    date: 'Today',
    ocrSnippet: 'Tcl_AsyncDelete is called when Tkinter interpreter is deallocated by a thread other than the creator thread.',
  },
  {
    id: 'act-010',
    application: 'Visual Studio Code',
    windowTitle: 'test_ui.py - BLINDSPOT Tests',
    category: 'Development',
    startTime: '11:15:00',
    endTime: '12:30:00',
    durationSeconds: 4500,
    durationFormatted: '1h 15m',
    confidence: 3,
    date: 'Today',
    ocrSnippet: 'class TestBlindspotMainWindow(unittest.TestCase): def test_11_window_initialization(self):',
  },
  {
    id: 'act-011',
    application: 'Postman',
    windowTitle: 'Gemini v1beta Interactions API - Test Collection',
    category: 'Development',
    startTime: '10:00:00',
    endTime: '11:15:00',
    durationSeconds: 4500,
    durationFormatted: '1h 15m',
    confidence: 3,
    date: 'Yesterday',
    ocrSnippet: 'POST https://generativelanguage.googleapis.com/v1beta/interactions 200 OK 482ms',
  },
  {
    id: 'act-012',
    application: 'Visual Studio Code',
    windowTitle: 'verification.py - TaskVerifier Implementation',
    category: 'Development',
    startTime: '08:45:00',
    endTime: '10:00:00',
    durationSeconds: 4500,
    durationFormatted: '1h 15m',
    confidence: 3,
    date: 'Yesterday',
    ocrSnippet: 'class DefaultContextObserver: def observe_fresh(self) -> DesktopContext: return fresh_context',
  },
];

export const mockScreenContext = {
  application: 'Visual Studio Code',
  windowTitle: 'ui.py — BLINDSPOT — Visual Studio Code (v1.98.0)',
  category: 'Development',
  confidence: 3,
  capturedAt: 'Just now (18:14:02)',
  wordCount: 142,
  characterCount: 986,
  detectedKeywords: ['def', 'class', 'threading', 'queue', 'tk.Tk', 'ScrolledText', '_handle_chat_response', 'QuotaExceededError'],
  rawOcrText: `class BlindspotMainWindow:
    """Main desktop window for BLINDSPOT with dual-tab interaction."""

    def _handle_chat_response(self, res: Dict[str, Any]) -> None:
        """Process chat response on UI thread and restore controls."""
        if self._is_destroyed:
            return

        try:
            if res.get("success"):
                answer = res.get("answer", "")
                if "[Local Fallback" in answer or "[Gemini Quota Exceeded]" in answer:
                    self._append_chat_message("BLINDSPOT (Local Fallback)", answer, tag="assistant")
                    self.set_status_message("Notice: Gemini quota exceeded. Local fallback active.")
                else:
                    self._append_chat_message("BLINDSPOT", answer, tag="assistant")
                    self.set_status_message("BLINDSPOT replied.")
            else:
                err = res.get("error", "Unknown error")
                self._append_chat_message("ERROR", err, tag="error")
        finally:
            self._is_chat_busy = False
            self.chat_entry.configure(state="normal")
            self.send_chat_btn.configure(state="normal")`,
  mockCodeSnippet: `// Active file in editor: src/components/ui.py
export function handleChatDispatch(res: ChatResponse) {
  if (res.isFallback) {
    appendLocalNotice("Gemini Quota Exceeded — Local Fallback Active");
    setPerceptionEngineStatus("HEURISTIC_OFFLINE_MODE");
  }
  restoreInputControls();
}`,
};
