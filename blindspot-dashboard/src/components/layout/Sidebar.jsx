import React from 'react';
import {
  LayoutDashboard,
  Activity,
  ScanEye,
  History,
  Settings,
  Shield,
  Layers,
  Cpu,
  Brain,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

const navItems = [
  {
    id: 'overview',
    label: 'Overview',
    icon: LayoutDashboard,
    description: 'Summary metrics & charts',
  },
  {
    id: 'live',
    label: 'Live Activity',
    icon: Activity,
    description: 'Active window & timeline',
  },
  {
    id: 'context',
    label: 'Screen Context',
    icon: ScanEye,
    description: 'OCR & activity perception',
  },
  {
    id: 'history',
    label: 'Activity History',
    icon: History,
    description: 'Searchable context logs',
  },
  {
    id: 'intelligence',
    label: 'Activity Intelligence',
    icon: Brain,
    description: 'Summaries & diagnostics',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    description: 'Privacy & cadence',
  },
];

export const Sidebar = () => {
  const {
    currentPage,
    setCurrentPage,
    isMonitoring,
    toggleMonitoring,
    connectionStatus,
    isDemoMode,
    setIsDemoMode,
  } = useMonitoring();

  const isLiveConnected = connectionStatus === 'connected' && !isDemoMode;
  const isDisconnected = connectionStatus === 'disconnected' && !isDemoMode;

  return (
    <aside className="w-64 bg-background-secondary border-r border-border flex flex-col justify-between shrink-0 h-screen sticky top-0 select-none">
      <div>
        {/* Brand header */}
        <div className="h-16 border-b border-border/80 px-5 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shadow-sm shadow-indigo-950">
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-wide text-white">BLINDSPOT</span>
              <span className="text-[10px] font-semibold tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                PRO
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-none mt-0.5">Desktop Assistant</p>
          </div>
        </div>

        {/* Monitoring status pill */}
        <div className="px-4 py-3 border-b border-border/40 space-y-2">
          <div className="p-2.5 rounded-lg bg-background-elevated border border-border/60 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  isLiveConnected
                    ? isMonitoring
                      ? 'bg-emerald-400 animate-pulse'
                      : 'bg-amber-400'
                    : isDisconnected
                    ? 'bg-rose-500'
                    : 'bg-indigo-400 animate-pulse'
                }`}
              />
              <span className="text-xs font-medium text-slate-300">
                {isLiveConnected
                  ? isMonitoring
                    ? 'Perception Active'
                    : 'Perception Paused'
                  : isDisconnected
                  ? 'Backend Offline'
                  : 'Demo Simulation'}
              </span>
            </div>
            <span
              className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded border ${
                isLiveConnected
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  : isDisconnected
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
              }`}
            >
              {isLiveConnected ? 'LIVE ENGINE' : isDisconnected ? 'DISCONNECTED' : 'DEMO MODE'}
            </span>
          </div>

          {/* Engine control buttons: Pause/Resume & Demo Toggle */}
          <div className="flex items-center gap-1.5 pt-0.5">
            <button
              onClick={toggleMonitoring}
              disabled={isDisconnected}
              className={`flex-1 py-1 px-2 rounded text-[11px] font-medium border transition-colors ${
                isMonitoring
                  ? 'bg-background-elevated hover:bg-background-subtle text-slate-300 border-border'
                  : 'bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border-indigo-500/40'
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {isMonitoring ? 'Pause Perception' : 'Resume Perception'}
            </button>
            <button
              onClick={() => setIsDemoMode(!isDemoMode)}
              className={`py-1 px-2 rounded text-[11px] font-mono border transition-colors ${
                isDemoMode
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : 'bg-background-elevated hover:bg-background-subtle text-slate-400 border-border'
              }`}
              title="Toggle between Live Engine and Simulated Demo Mode"
            >
              {isDemoMode ? 'Exit Demo' : 'Demo Mode'}
            </button>
          </div>
        </div>

        {/* Navigation items */}
        <nav className="p-3 space-y-1">
          <div className="text-[10px] uppercase font-semibold text-slate-500 px-3 py-1.5 tracking-wider">
            Workspace
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all group ${
                  isActive
                    ? 'bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-background-elevated/70 border border-transparent'
                }`}
              >
                <Icon
                  className={`w-4 h-4 transition-colors ${
                    isActive ? 'text-indigo-400' : 'text-slate-400 group-hover:text-slate-300'
                  }`}
                />
                <div className="flex-1 text-left">
                  <div className="leading-tight">{item.label}</div>
                </div>
                {isActive && (
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer system status */}
      <div className="p-3 border-t border-border/70 space-y-2">
        <div className="p-3 rounded-lg bg-background-elevated/60 border border-border/50">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              Engine
            </span>
            <span className="font-mono text-[11px] text-slate-300">Phase 9A</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              Privacy Mode
            </span>
            <span className="text-[11px] text-emerald-400 font-medium">Local-Only</span>
          </div>
        </div>

        <div className="text-[10px] text-slate-500 text-center py-1 flex items-center justify-center gap-1">
          {isLiveConnected ? (
            <>
              <Wifi className="w-3 h-3 text-emerald-400" />
              <span>Connected: 127.0.0.1:8000</span>
            </>
          ) : isDisconnected ? (
            <>
              <WifiOff className="w-3 h-3 text-rose-400" />
              <span>Offline: 127.0.0.1:8000</span>
            </>
          ) : (
            <span>Demo Mode • Mock Data</span>
          )}
        </div>
      </div>
    </aside>
  );
};
