import React from 'react';
import { useMonitoring } from '../../context/MonitoringContext';
import { Badge } from '../common/Badge';
import {
  Monitor,
  Sparkles,
  Activity,
} from 'lucide-react';

export const ActiveWindowCard = () => {
  const {
    liveContext,
    isMonitoring,
    switchSimulatedApp,
    connectionStatus,
    isDemoMode,
  } = useMonitoring();

  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const formatDuration = (seconds) => {
    const sec = Math.max(0, parseInt(seconds, 10) || 0);
    const hours = Math.floor(sec / 3600);
    const minutes = Math.floor((sec % 3600) / 60);
    const s = sec % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${s}s`;
    }
    return `${minutes}m ${s}s`;
  };

  const simulatedTargets = [
    { app: 'Visual Studio Code', title: 'ui.py — BLINDSPOT — Visual Studio Code', cat: 'Development' },
    { app: 'Google Chrome', title: 'Gemini 1.5 Pro Docs & Quota Limits', cat: 'Research' },
    { app: 'Slack', title: '#general — Engineering Standup — Slack', cat: 'Communication' },
    { app: 'Windows Terminal', title: 'PowerShell - python main.py --ui', cat: 'Development' },
  ];

  const bounds = liveContext.windowBounds || { width: 1920, height: 1080 };

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5">
      {/* Top Banner Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border/70 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-base font-semibold text-white">Active Foreground Window</h3>
            <Badge
              variant={isMonitoring ? 'success' : 'neutral'}
              size="sm"
              pulse={isMonitoring}
            >
              {isMonitoring ? (isLive ? 'OBSERVING REAL OS' : 'OBSERVING DEMO') : 'PAUSED'}
            </Badge>
          </div>
          <p className="text-xs text-slate-400">
            {isLive
              ? 'Real Win32 GetForegroundWindow hook observation stream'
              : 'Simulated desktop window observation'}
          </p>
        </div>

        {/* Pulse Heartbeat */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background-elevated border border-border text-xs text-slate-300">
            <Activity className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
            <span className="font-mono text-[11px]">{isLive ? 'Engine Hook 1.5s' : 'Local Timer'}</span>
          </div>
        </div>
      </div>

      {/* Main Focus Banner */}
      <div className="mt-6 p-4 rounded-xl bg-background-elevated/70 border border-border/80">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 shrink-0">
            <Monitor className="w-6 h-6" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-base font-bold text-white truncate">
                {liveContext.activeApplication || 'None'}
              </span>
              <span className="text-xs font-mono text-slate-400 bg-background-secondary px-2 py-0.5 rounded border border-border/50">
                PID: {liveContext.processId || 'N/A'}
              </span>
              <Badge variant="primary" size="sm">
                {liveContext.category || 'Productivity'}
              </Badge>
            </div>

            <div className="text-xs font-mono text-indigo-300 bg-background-secondary/90 px-3 py-2 rounded-lg border border-border/50 truncate mb-3">
              {liveContext.activeWindow || 'No active window detected'}
            </div>

            {/* Metrics grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-background-secondary border border-border/40">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Active Focus</span>
                <span className="font-mono text-sm font-bold text-emerald-400">
                  {formatDuration(liveContext.currentDurationSeconds)}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-background-secondary border border-border/40">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Confidence</span>
                <span className="font-mono text-sm font-bold text-indigo-300">
                  {liveContext.confidence || 0} / 3 {liveContext.confidence === 3 ? 'High' : liveContext.confidence === 2 ? 'Medium' : 'Low'}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-background-secondary border border-border/40">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Geometry</span>
                <span className="font-mono text-sm text-slate-300">
                  {bounds.width || 1920} × {bounds.height || 1080}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-background-secondary border border-border/40">
                <span className="text-slate-400 block text-[10px] uppercase font-semibold">Pattern</span>
                <span className="font-mono text-sm text-amber-400 truncate block">
                  {liveContext.workflowPattern || 'focused_session'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Heuristic Interpretation */}
      <div className="mt-4 p-3.5 rounded-lg bg-background-elevated/40 border border-border/40 flex items-start gap-3">
        <Sparkles className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <div className="text-xs">
          <span className="font-semibold text-slate-200">Heuristic Workflow Interpretation: </span>
          <span className="text-slate-400">{liveContext.workflowInterpretation || 'Observation active.'}</span>
        </div>
      </div>

      {/* Manual Focus Simulation Controls */}
      <div className="mt-5 pt-4 border-t border-border/40">
        <div className="text-xs font-semibold text-slate-300 mb-2 flex items-center justify-between">
          <span>{isLive ? 'Real Windows Focus Detection' : 'Simulate Desktop Window Change'}</span>
          <span className="text-[11px] text-slate-400">
            {isLive ? 'Real-time Win32 foreground observation' : 'Click to test instant state transition'}
          </span>
        </div>
        {isLive ? (
          <div className="p-3 rounded-lg bg-background-elevated/40 border border-border/40 text-xs text-slate-400 flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Foreground window switch is tracked automatically whenever you switch apps on your Windows desktop.</span>
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {simulatedTargets.map((item) => (
              <button
                key={item.app}
                onClick={() => switchSimulatedApp(item.app, item.title, item.cat)}
                className={`p-2 rounded-lg text-xs text-left border transition-all ${
                  liveContext.activeApplication === item.app
                    ? 'bg-indigo-600/20 border-indigo-500/50 text-indigo-300 font-medium'
                    : 'bg-background-elevated/60 hover:bg-background-elevated border-border/50 text-slate-300'
                }`}
              >
                <div className="font-semibold truncate">{item.app}</div>
                <div className="text-[10px] text-slate-400 font-mono truncate">{item.cat}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
