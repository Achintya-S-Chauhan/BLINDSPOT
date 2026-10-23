import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  Clock,
  Monitor,
  ChevronDown,
} from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';
import { Badge } from '../common/Badge';

export const Header = () => {
  const {
    currentPage,
    isMonitoring,
    toggleMonitoring,
    liveContext,
    switchSimulatedApp,
    connectionStatus,
    isDemoMode,
  } = useMonitoring();

  const [time, setTime] = useState('');
  const [date, setDate] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const updateDateTime = () => {
      const now = new Date();
      setTime(
        now.toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      );
      setDate(
        now.toLocaleDateString([], {
          weekday: 'short',
          month: 'short',
          day: 'numeric',
          year: 'numeric',
        })
      );
    };

    updateDateTime();
    const interval = setInterval(updateDateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const getPageTitle = (page) => {
    switch (page) {
      case 'overview':
        return 'Overview Dashboard';
      case 'live':
        return 'Live Desktop Activity';
      case 'context':
        return 'Screen Context & OCR';
      case 'history':
        return 'Historical Activity Logs';
      case 'intelligence':
        return 'Activity Intelligence & Diagnostics';
      case 'settings':
        return 'Perception & Privacy Settings';
      default:
        return 'Dashboard';
    }
  };

  const simulationPresets = [
    { app: 'Visual Studio Code', title: 'ui.py — BLINDSPOT — Visual Studio Code', cat: 'Development' },
    { app: 'Google Chrome', title: 'Gemini API Reference — Google Docs', cat: 'Research' },
    { app: 'Slack', title: '#general — Engineering Standup — Slack', cat: 'Communication' },
    { app: 'Windows Terminal', title: 'PowerShell - python main.py --ui', cat: 'Development' },
  ];

  const isDisconnected = connectionStatus === 'disconnected' && !isDemoMode;

  return (
    <header className="h-16 border-b border-border/80 bg-background-secondary/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Current Page Title & Subtitle */}
      <div>
        <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
          {getPageTitle(currentPage)}
        </h1>
        <p className="text-[11px] text-slate-400">
          Context-Aware Desktop Perception Engine
        </p>
      </div>

      {/* Header Actions & Telemetry */}
      <div className="flex items-center gap-4">
        {/* Fast Focus Switcher / Current App Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background-elevated hover:bg-background-elevated/80 border border-border text-xs text-slate-200 transition-colors"
          >
            <Monitor className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-medium max-w-[120px] sm:max-w-[160px] truncate">
              {liveContext.activeApplication || 'No Active App'}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-72 rounded-xl bg-background-secondary border border-border shadow-2xl p-2 z-50">
              <div className="text-[10px] uppercase font-semibold text-slate-400 px-2 py-1 tracking-wider flex items-center justify-between">
                <span>{isDemoMode ? 'Simulate Window Switch' : 'Current Active Window'}</span>
                <span className={`text-[9px] px-1.5 py-0.2 rounded font-mono ${isDemoMode ? 'bg-amber-500/20 text-amber-300' : 'bg-emerald-500/20 text-emerald-300'}`}>
                  {isDemoMode ? 'DEMO' : 'LIVE WIN32'}
                </span>
              </div>

              {!isDemoMode ? (
                <div className="p-2 space-y-1.5 text-xs text-slate-300">
                  <div className="text-[11px] font-medium text-white truncate">
                    {liveContext.activeWindow || 'No active foreground window'}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    PID: {liveContext.processId || 'N/A'} • {liveContext.category}
                  </div>
                  <p className="text-[10px] text-slate-400 pt-1 border-t border-border/50">
                    Live telemetry is read from the Windows OS foreground hook. Switch to Demo Mode in the sidebar to simulate manual window switches.
                  </p>
                </div>
              ) : (
                <div className="space-y-1">
                  {simulationPresets.map((preset) => (
                    <button
                      key={preset.app}
                      onClick={() => {
                        switchSimulatedApp(preset.app, preset.title, preset.cat);
                        setIsDropdownOpen(false);
                      }}
                      className={`w-full text-left p-2 rounded-lg text-xs transition-colors flex flex-col ${
                        liveContext.activeApplication === preset.app
                          ? 'bg-indigo-600/20 text-indigo-300 font-medium'
                          : 'hover:bg-background-elevated text-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{preset.app}</span>
                        <span className="text-[10px] text-slate-400 font-mono">{preset.cat}</span>
                      </div>
                      <span className="text-[11px] text-slate-400 truncate mt-0.5">
                        {preset.title}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Date & Time Display */}
        <div className="hidden lg:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-background-elevated/50 border border-border/50 text-xs text-slate-300 font-mono">
          <Clock className="w-3.5 h-3.5 text-slate-400" />
          <span>{date}</span>
          <span className="text-slate-500">•</span>
          <span className="text-indigo-400 font-semibold">{time}</span>
        </div>

        {/* Monitoring Toggle Button */}
        <button
          onClick={toggleMonitoring}
          disabled={isDisconnected}
          title={isDisconnected ? 'BLINDSPOT engine backend is offline' : ''}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs transition-all shadow-sm ${
            isDisconnected
              ? 'bg-slate-800 text-slate-500 border border-slate-700 opacity-60 cursor-not-allowed'
              : isMonitoring
              ? 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30'
              : 'bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
          }`}
        >
          {isMonitoring ? (
            <>
              <Pause className="w-3.5 h-3.5 text-rose-400" />
              <span>Pause Perception</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 text-emerald-400" />
              <span>Resume Perception</span>
            </>
          )}
        </button>

        {/* Badge Indicator */}
        {isDemoMode ? (
          <Badge variant="warning" size="sm">
            DEMO MODE
          </Badge>
        ) : isDisconnected ? (
          <Badge variant="danger" size="sm">
            DISCONNECTED
          </Badge>
        ) : isMonitoring ? (
          <Badge variant="success" size="sm" pulse={true}>
            ONLINE
          </Badge>
        ) : (
          <Badge variant="neutral" size="sm">
            PAUSED
          </Badge>
        )}
      </div>
    </header>
  );
};
