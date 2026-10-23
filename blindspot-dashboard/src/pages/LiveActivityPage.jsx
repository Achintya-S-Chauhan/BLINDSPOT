import React from 'react';
import { ActiveWindowCard } from '../components/live/ActiveWindowCard';
import { RecentSwitchesList } from '../components/live/RecentSwitchesList';
import { Info, Wifi, WifiOff } from 'lucide-react';
import { useMonitoring } from '../context/MonitoringContext';

export const LiveActivityPage = () => {
  const { connectionStatus, isDemoMode } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;
  const isDisconnected = connectionStatus === 'disconnected' && !isDemoMode;

  return (
    <div className="space-y-6">
      {/* Real / Disconnected / Simulation Status Banner */}
      <div
        className={`p-3.5 rounded-xl border flex items-center justify-between text-xs ${
          isLive
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
            : isDisconnected
            ? 'bg-rose-500/10 border-rose-500/20 text-rose-300'
            : 'bg-background-secondary border-border/70 text-slate-300'
        }`}
      >
        <div className="flex items-center gap-2">
          {isLive ? (
            <Wifi className="w-4 h-4 text-emerald-400 shrink-0" />
          ) : isDisconnected ? (
            <WifiOff className="w-4 h-4 text-rose-400 shrink-0" />
          ) : (
            <Info className="w-4 h-4 text-indigo-400 shrink-0" />
          )}
          <span>
            {isLive ? (
              <>
                <strong>Live Win32 Desktop Feed:</strong> Real-time foreground window detection is running with native Windows hooks (127.0.0.1:8000).
              </>
            ) : isDisconnected ? (
              <>
                <strong>Backend Disconnected:</strong> Cannot receive foreground window events. Verify BLINDSPOT backend is running at <code className="bg-background-elevated px-1 py-0.5 rounded text-rose-200">127.0.0.1:8000</code>.
              </>
            ) : (
              <>
                <strong>Simulated Desktop Feed:</strong> Operating on client mock state. Switch targets using the buttons below or click Exit Demo in the sidebar.
              </>
            )}
          </span>
        </div>
      </div>

      <ActiveWindowCard />

      <RecentSwitchesList />
    </div>
  );
};
