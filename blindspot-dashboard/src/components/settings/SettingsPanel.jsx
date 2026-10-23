import React, { useState } from 'react';
import { useMonitoring } from '../../context/MonitoringContext';
import {
  Shield,
  Clock,
  Plus,
  X,
  Lock,
  HardDrive,
  Check,
  AlertCircle,
} from 'lucide-react';

export const SettingsPanel = () => {
  const {
    settings,
    updateSettings,
    addExcludedApp,
    removeExcludedApp,
    connectionStatus,
    isDemoMode,
  } = useMonitoring();

  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const [newAppInput, setNewAppInput] = useState('');
  const [saveToast, setSaveToast] = useState(false);

  const handleAddApp = (e) => {
    e.preventDefault();
    if (newAppInput.trim()) {
      addExcludedApp(newAppInput.trim());
      setNewAppInput('');
    }
  };

  const handleSaveNotification = () => {
    setSaveToast(true);
    setTimeout(() => setSaveToast(false), 2500);
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className={`p-4 rounded-xl border text-xs flex items-start gap-3 ${
        isLive
          ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
          : 'bg-amber-500/10 border-amber-500/20 text-amber-300'
      }`}>
        <AlertCircle className={`w-4 h-4 shrink-0 mt-0.5 ${isLive ? 'text-emerald-400' : 'text-amber-400'}`} />
        <div>
          <span className="font-semibold block mb-0.5">
            {isLive ? 'Live Engine Runtime Parameters:' : 'Simulated Settings Context:'}
          </span>
          {isLive ? (
            <>
              Operating with active BLINDSPOT Python perception parameters: Window Polling: <strong className="text-white">300ms</strong>, Stability Threshold: <strong className="text-white">1500ms</strong>, Periodic OCR: <strong className="text-white">3.0s</strong>, In-Memory Ring Buffer: <strong className="text-white">50 episodes</strong>, I/O Backend: <strong className="text-white">{settings.backendIo || 'WindowsDesktopIO'}</strong>. Core timing loops are enforced in memory by the backend.
            </>
          ) : (
            <>
              These configuration controls update reactive dashboard state in memory for testing and prototype demonstrations. In live mode, runtime parameters are governed directly by the active BLINDSPOT daemon.
            </>
          )}
        </div>
      </div>

      {/* Grid of Settings sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Cadence & Frequency Card */}
        <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-semibold text-white">Observation Cadence</h3>
            </div>

            {/* Poll interval slider */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Window Polling Rate</span>
                  <span className="font-mono text-indigo-300 font-semibold">
                    {settings.pollIntervalMs} ms
                  </span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="2000"
                  step="100"
                  value={settings.pollIntervalMs}
                  onChange={(e) => updateSettings({ pollIntervalMs: Number(e.target.value) })}
                  className="w-full accent-indigo-500 h-1.5 bg-background-elevated rounded-lg cursor-pointer"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  How often BLINDSPOT queries Win32 GetForegroundWindow (Default: 500ms).
                </p>
              </div>

              {/* OCR cadence slider */}
              <div className="pt-3 border-t border-border/40">
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Periodic OCR Cadence</span>
                  <span className="font-mono text-indigo-300 font-semibold">
                    Every {settings.ocrIntervalSeconds} s
                  </span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="60"
                  step="5"
                  value={settings.ocrIntervalSeconds}
                  onChange={(e) => updateSettings({ ocrIntervalSeconds: Number(e.target.value) })}
                  className="w-full accent-indigo-500 h-1.5 bg-background-elevated rounded-lg cursor-pointer"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Frequency of optical keyword scans across active foreground rectangle.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-border/40 text-[11px] text-slate-500">
            Higher polling rates increase CPU utilization slightly.
          </div>
        </div>

        {/* Privacy & Local Storage */}
        <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-white">Privacy & Local Storage</h3>
            </div>

            <div className="space-y-4 text-xs">
              {/* RAM Only Toggle */}
              <div className="flex items-center justify-between p-3 rounded-lg bg-background-elevated/40 border border-border/50">
                <div>
                  <div className="font-medium text-slate-200">Strict RAM-Only Memory</div>
                  <div className="text-[11px] text-slate-400">Never write window history to disk</div>
                </div>
                <button
                  onClick={() => updateSettings({ inMemoryOnly: !settings.inMemoryOnly })}
                  className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors ${
                    settings.inMemoryOnly ? 'bg-indigo-600' : 'bg-slate-700'
                  }`}
                >
                  <div
                    className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                      settings.inMemoryOnly ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Blur sensitive fields */}
              <div className="flex items-center justify-between p-3 rounded-lg bg-background-elevated/40 border border-border/50">
                <div>
                  <div className="font-medium text-slate-200">Mask Passwords & API Secrets</div>
                  <div className="text-[11px] text-slate-400">Regex redaction on OCR text buffers</div>
                </div>
                <button
                  onClick={() => updateSettings({ blurSensitiveFields: !settings.blurSensitiveFields })}
                  className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors ${
                    settings.blurSensitiveFields ? 'bg-indigo-600' : 'bg-slate-700'
                  }`}
                >
                  <div
                    className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                      settings.blurSensitiveFields ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>

              {/* Data retention */}
              <div className="flex items-center justify-between p-3 rounded-lg bg-background-elevated/40 border border-border/50">
                <div>
                  <div className="font-medium text-slate-200">Activity Retention</div>
                  <div className="text-[11px] text-slate-400">Automated local cache purge</div>
                </div>
                <span className="font-mono text-indigo-300 font-semibold px-2 py-1 bg-background-secondary rounded border border-border/50">
                  {settings.retentionDays} Days
                </span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-border/40 text-[11px] text-slate-500 flex items-center gap-1">
            <Lock className="w-3.5 h-3.5 text-slate-400" />
            Zero external cloud data synchronization.
          </div>
        </div>
      </div>

      {/* Excluded Applications List */}
      <div className="bg-background-secondary border border-border rounded-xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-semibold text-white">Application Exclusion Filters</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Processes that BLINDSPOT automatically ignores when in foreground.
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-400 bg-background-elevated px-2 py-1 rounded border border-border/50">
            {settings.excludedApps.length} Excluded
          </span>
        </div>

        {/* Input to add application */}
        <form onSubmit={handleAddApp} className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="e.g. Keepass.exe, BankingPortal"
            value={newAppInput}
            onChange={(e) => setNewAppInput(e.target.value)}
            className="flex-1 max-w-sm px-3 py-1.5 bg-background-elevated border border-border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/60"
          />
          <button
            type="submit"
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add Process</span>
          </button>
        </form>

        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {settings.excludedApps.map((app) => (
            <span
              key={app}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-background-elevated border border-border text-xs text-slate-300 font-mono"
            >
              <HardDrive className="w-3 h-3 text-slate-500" />
              <span>{app}</span>
              <button
                type="button"
                onClick={() => removeExcludedApp(app)}
                className="hover:text-rose-400 p-0.5 rounded transition-colors ml-1"
                title={`Remove ${app}`}
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      </div>

      {/* Save Settings Trigger */}
      <div className="flex items-center justify-between p-4 rounded-xl bg-background-secondary border border-border">
        <span className="text-xs text-slate-400">
          Preferences auto-persist to active prototype state.
        </span>
        <button
          onClick={handleSaveNotification}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-md shadow-indigo-950"
        >
          {saveToast ? (
            <>
              <Check className="w-4 h-4 text-emerald-300" />
              <span>Settings Saved!</span>
            </>
          ) : (
            <span>Apply Changes</span>
          )}
        </button>
      </div>
    </div>
  );
};
