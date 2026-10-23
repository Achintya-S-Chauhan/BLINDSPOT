import React from 'react';
import { useMonitoring } from '../../context/MonitoringContext';
import { ArrowRight, Clock } from 'lucide-react';

export const RecentSwitchesList = () => {
  const { liveContext } = useMonitoring();

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">Application Switch History</h3>
          <p className="text-xs text-slate-400 mt-0.5">Rapid context transitions recorded this session</p>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-background-elevated px-2 py-1 rounded border border-border/50">
          {liveContext.recentSwitches.length} Switches
        </span>
      </div>

      <div className="space-y-2.5">
        {(liveContext.recentSwitches || []).length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-400 bg-background-elevated/30 rounded-lg border border-border/40">
            No application transitions recorded yet. As you switch across Windows applications, transitions will be cataloged here in real time.
          </div>
        ) : (
          (liveContext.recentSwitches || []).map((item) => (
            <div
              key={item.id}
              className="p-3 rounded-lg bg-background-elevated/40 hover:bg-background-elevated border border-border/40 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-2"
            >
              <div className="flex items-center gap-2.5">
                <span className="text-xs font-semibold text-white px-2.5 py-1 rounded bg-background-secondary border border-border/60">
                  {item.fromApp}
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                <span className="text-xs font-semibold text-indigo-300 px-2.5 py-1 rounded bg-indigo-600/15 border border-indigo-500/30">
                  {item.toApp}
                </span>
              </div>

              <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3 text-slate-500" />
                  {item.timestamp}
                </span>
                <span className="text-slate-500">•</span>
                <span className="text-emerald-400 font-medium">Spent {item.durationInApp}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
