import React from 'react';

export const StatCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  badgeText,
  badgeVariant = 'neutral',
}) => {
  return (
    <div className="bg-background-secondary border border-border/70 hover:border-border-accent/80 transition-colors duration-200 rounded-xl p-5 flex flex-col justify-between relative overflow-hidden group">
      {/* Top subtle glow line on hover */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-indigo-500/0 to-transparent group-hover:via-indigo-500/40 transition-all duration-300" />

      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            {title}
          </span>
          <div className="p-2 rounded-lg bg-background-elevated border border-border/50 text-slate-300 group-hover:text-indigo-300 transition-colors">
            <Icon className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-baseline gap-2">
          <div className="text-2xl font-bold tracking-tight text-white font-mono">
            {value}
          </div>
          {badgeText && (
            <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${
              badgeVariant === 'success'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20'
            }`}>
              {badgeText}
            </span>
          )}
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-border/40 flex items-center justify-between text-xs text-slate-400">
        <span>{subtitle}</span>
        {trend && (
          <span
            className={`font-medium flex items-center gap-0.5 ${
              trend.isPositive ? 'text-emerald-400' : 'text-slate-400'
            }`}
          >
            {trend.value}
            {trend.label && <span className="text-slate-500 ml-1">{trend.label}</span>}
          </span>
        )}
      </div>
    </div>
  );
};
