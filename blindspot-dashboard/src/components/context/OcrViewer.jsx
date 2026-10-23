import React, { useState } from 'react';
import { Search, Copy, Check, Tag, Sparkles } from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const OcrViewer = ({ contextData = {}, isLoading }) => {
  const { connectionStatus, isDemoMode } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const [searchTerm, setSearchTerm] = useState('');
  const [copied, setCopied] = useState(false);

  const rawText = contextData?.rawOcrText || '';
  const keywords = contextData?.detectedKeywords || [];

  const handleCopy = () => {
    navigator.clipboard.writeText(rawText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const filteredText = searchTerm
    ? rawText
        .split('\n')
        .filter((line) => line.toLowerCase().includes(searchTerm.toLowerCase()))
        .join('\n')
    : rawText;

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">OCR Inferred Text Stream</h3>
            <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
              isLive
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20'
            }`}>
              {isLive ? 'Tesseract OCR Live' : 'Tesseract Engine Mock'}
            </span>
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-background-elevated hover:bg-background-elevated/80 border border-border text-xs text-slate-300 transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy OCR</span>
              </>
            )}
          </button>
        </div>

        {/* Stats bar & Search */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
            <span>Words: <strong className="text-white">{contextData?.wordCount ?? rawText.split(/\s+/).filter(Boolean).length}</strong></span>
            <span>Chars: <strong className="text-white">{contextData?.characterCount ?? rawText.length}</strong></span>
          </div>

          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search in OCR buffer..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full sm:w-48 pl-8 pr-3 py-1 bg-background-elevated border border-border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/60"
            />
          </div>
        </div>

        {/* Keywords detected */}
        <div className="mb-3 flex flex-wrap gap-1.5">
          {keywords.map((kw) => (
            <span
              key={kw}
              className="text-[10px] font-mono px-2 py-0.5 rounded bg-background-elevated border border-border/60 text-slate-300 flex items-center gap-1"
            >
              <Tag className="w-2.5 h-2.5 text-indigo-400" />
              {kw}
            </span>
          ))}
        </div>

        {/* Text Viewport */}
        <div className="p-3.5 rounded-lg bg-[#0c0d15] border border-border/80 font-mono text-xs text-slate-300 h-64 overflow-y-auto leading-relaxed relative">
          {isLoading ? (
            <div className="flex items-center justify-center h-full text-slate-500">
              Processing OCR buffer...
            </div>
          ) : filteredText.length > 0 ? (
            <pre className="whitespace-pre-wrap">{filteredText}</pre>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-500 text-center">
              No matching keywords found in current OCR buffer.
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-border/40 text-xs text-slate-400 flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          OCR payload sanitized & truncated to prevent memory bloat.
        </span>
        <span className="font-mono text-[11px] text-slate-400">Limit: 250 chars</span>
      </div>
    </div>
  );
};
