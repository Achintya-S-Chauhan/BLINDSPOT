import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import { mockHourlyActivity, mockAppUsage } from '../../data/mockData';
import { BarChart3, LineChart, Info } from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const UsageChart = () => {
  const { overviewData, isDemoMode, connectionStatus } = useMonitoring();
  const [viewType, setViewType] = useState('apps');

  const isLive = connectionStatus === 'connected' && !isDemoMode;

  // Real app usage or demo data
  const appData = (isLive && overviewData?.appUsage?.length > 0)
    ? overviewData.appUsage
    : mockAppUsage;

  const hourlyData = (isLive && overviewData?.hourlyActivity?.length > 0)
    ? overviewData.hourlyActivity
    : mockHourlyActivity;

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
      {/* Chart Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">Application Usage & Time Flow</h3>
            <span className="text-[11px] text-slate-400 font-mono">
              {isLive ? 'Live Engine' : 'Simulated'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {viewType === 'hourly'
              ? 'Activity category distribution across working hours (in minutes)'
              : 'Cumulative focus time spent per detected application'}
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex items-center bg-background-elevated p-1 rounded-lg border border-border/80 self-start sm:self-auto">
          <button
            onClick={() => setViewType('apps')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
              viewType === 'apps'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <LineChart className="w-3.5 h-3.5" />
            <span>Top Apps</span>
          </button>
          <button
            onClick={() => setViewType('hourly')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
              viewType === 'hourly'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Hourly Flow</span>
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {viewType === 'hourly' ? (
            <BarChart data={hourlyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2233" vertical={false} />
              <XAxis dataKey="hour" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="m" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11131d',
                  borderColor: '#26293d',
                  borderRadius: '0.75rem',
                  fontSize: '12px',
                  color: '#f8fafc',
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                }}
                itemStyle={{ padding: '2px 0' }}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                iconType="circle"
              />
              <Bar dataKey="development" name="Development" stackId="a" fill="#6366f1" radius={[0, 0, 0, 0]} />
              <Bar dataKey="research" name="Research" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} />
              <Bar dataKey="communication" name="Communication" stackId="a" fill="#10b981" radius={[0, 0, 0, 0]} />
              <Bar dataKey="documentation" name="Documentation" stackId="a" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          ) : (
            <AreaChart
              data={appData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 40, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2233" horizontal={false} />
              <XAxis type="number" stroke="#64748b" fontSize={11} unit="s" tickFormatter={(v) => `${Math.round(v / 3600)}h`} />
              <YAxis type="category" dataKey="name" stroke="#cbd5e1" fontSize={11} tickLine={false} width={130} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#11131d',
                  borderColor: '#26293d',
                  borderRadius: '0.75rem',
                  fontSize: '12px',
                  color: '#f8fafc',
                }}
                formatter={(value) => [
                  `${Math.floor(Number(value) / 3600)}h ${Math.floor((Number(value) % 3600) / 60)}m`,
                  'Focus Time',
                ]}
              />
              <Area type="monotone" dataKey="totalSeconds" stroke="#818cf8" fill="#6366f1" fillOpacity={0.25} />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>

      <div className="mt-4 pt-3 border-t border-border/40 flex items-center justify-between text-xs text-slate-400">
        <span className="flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-indigo-400" />
          Metrics reflect non-idle foreground window presence.
        </span>
        <span className="text-slate-400 font-mono text-[11px]">Sampling: 500ms</span>
      </div>
    </div>
  );
};
