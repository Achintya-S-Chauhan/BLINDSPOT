import React, { useState, useMemo } from 'react';
import { mockActivityHistory } from '../../data/mockData';
import { Badge } from '../common/Badge';
import { EmptyState } from '../common/EmptyState';
import {
  Search,
  Download,
  Code2,
  Globe,
  MessageSquare,
  FileText,
  ChevronLeft,
  ChevronRight,
  Inbox,
} from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const ActivityTable = () => {
  const { activityHistory, connectionStatus, isDemoMode } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedDate, setSelectedDate] = useState('All');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 6;

  const rawRecords = activityHistory?.records;
  const rawHistoryList = useMemo(() => {
    return isLive ? (rawRecords || []) : mockActivityHistory;
  }, [isLive, rawRecords]);

  // Filter logic
  const filteredData = useMemo(() => {
    return rawHistoryList.filter((item) => {
      const matchesSearch =
        (item.application || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.windowTitle || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.ocrSnippet && item.ocrSnippet.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesCategory =
        selectedCategory === 'All' || item.category === selectedCategory;

      const matchesDate =
        selectedDate === 'All' || item.date === selectedDate;

      return matchesSearch && matchesCategory && matchesDate;
    });
  }, [rawHistoryList, searchTerm, selectedCategory, selectedDate]);

  // Pagination logic
  const totalPages = Math.ceil(filteredData.length / itemsPerPage) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredData.slice(start, start + itemsPerPage);
  }, [filteredData, currentPage, itemsPerPage]);

  const handleExportCSV = () => {
    const headers = 'ID,Application,WindowTitle,Category,StartTime,EndTime,Duration,Date\n';
    const rows = filteredData
      .map(
        (r) =>
          `"${r.id}","${r.application}","${r.windowTitle.replace(/"/g, '""')}","${r.category}","${r.startTime}","${r.endTime}","${r.durationFormatted}","${r.date}"`
      )
      .join('\n');

    const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `blindspot-activity-${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getCategoryBadgeVariant = (cat) => {
    switch (cat) {
      case 'Development':
        return 'primary';
      case 'Research':
        return 'info';
      case 'Communication':
        return 'success';
      case 'Documentation':
        return 'warning';
      case 'Design':
        return 'danger';
      default:
        return 'neutral';
    }
  };

  const getCategoryIcon = (cat) => {
    switch (cat) {
      case 'Development':
        return <Code2 className="w-3.5 h-3.5 text-indigo-400" />;
      case 'Research':
        return <Globe className="w-3.5 h-3.5 text-blue-400" />;
      case 'Communication':
        return <MessageSquare className="w-3.5 h-3.5 text-emerald-400" />;
      case 'Documentation':
        return <FileText className="w-3.5 h-3.5 text-amber-400" />;
      default:
        return <Code2 className="w-3.5 h-3.5 text-slate-400" />;
    }
  };

  return (
    <div className="bg-background-secondary border border-border rounded-xl overflow-hidden flex flex-col justify-between">
      {/* Table Filter Toolbar */}
      <div className="p-4 border-b border-border/80 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
        <div className="flex flex-1 items-center gap-3">
          {/* Search Bar */}
          <div className="relative flex-1 max-w-sm">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search application, window title, or OCR text..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full pl-9 pr-3 py-1.5 bg-background-elevated border border-border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/60"
            />
          </div>

          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => {
              setSelectedCategory(e.target.value);
              setCurrentPage(1);
            }}
            className="px-3 py-1.5 bg-background-elevated border border-border rounded-lg text-xs text-slate-200 focus:outline-none focus:border-indigo-500/60"
          >
            <option value="All">All Categories</option>
            <option value="Development">Development</option>
            <option value="Research">Research</option>
            <option value="Communication">Communication</option>
            <option value="Documentation">Documentation</option>
            <option value="Design">Design</option>
          </select>

          {/* Date Filter */}
          <select
            value={selectedDate}
            onChange={(e) => {
              setSelectedDate(e.target.value);
              setCurrentPage(1);
            }}
            className="px-3 py-1.5 bg-background-elevated border border-border rounded-lg text-xs text-slate-200 focus:outline-none focus:border-indigo-500/60"
          >
            <option value="All">All Dates</option>
            <option value="Today">Today</option>
            <option value="Yesterday">Yesterday</option>
          </select>
        </div>

        {/* Export action */}
        <button
          onClick={handleExportCSV}
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-background-elevated hover:bg-background-elevated/80 border border-border text-xs text-slate-300 hover:text-white transition-colors"
        >
          <Download className="w-3.5 h-3.5 text-slate-400" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto min-h-[300px]">
        {paginatedData.length > 0 ? (
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-[#121420] border-b border-border text-[11px] uppercase tracking-wider text-slate-400 font-semibold select-none">
              <tr>
                <th className="py-3 px-4">Application</th>
                <th className="py-3 px-4">Active Window Title</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Time Interval</th>
                <th className="py-3 px-4">Duration</th>
                <th className="py-3 px-4 text-center">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {paginatedData.map((row) => (
                <tr
                  key={row.id}
                  className="hover:bg-background-elevated/50 transition-colors"
                >
                  <td className="py-3 px-4 font-semibold text-white whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      {getCategoryIcon(row.category)}
                      <span>{row.application}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 font-mono text-xs max-w-xs truncate text-slate-300">
                    <span title={row.windowTitle}>{row.windowTitle}</span>
                  </td>
                  <td className="py-3 px-4 whitespace-nowrap">
                    <Badge variant={getCategoryBadgeVariant(row.category)} size="sm">
                      {row.category}
                    </Badge>
                  </td>
                  <td className="py-3 px-4 whitespace-nowrap font-mono text-[11px] text-slate-400">
                    {row.startTime} - {row.endTime}
                  </td>
                  <td className="py-3 px-4 whitespace-nowrap font-mono font-medium text-emerald-400">
                    {row.durationFormatted}
                  </td>
                  <td className="py-3 px-4 text-center whitespace-nowrap font-mono text-slate-400">
                    <span className="px-2 py-0.5 rounded bg-background-elevated border border-border/60 text-[11px] text-indigo-300">
                      {row.confidence}/3
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="p-8">
            <EmptyState
              icon={Inbox}
              title="No activity records match your criteria"
              description="Try adjusting your search query, selecting another category, or resetting the date filter."
              actionText="Reset Filters"
              onAction={() => {
                setSearchTerm('');
                setSelectedCategory('All');
                setSelectedDate('All');
              }}
            />
          </div>
        )}
      </div>

      {/* Pagination Footer */}
      <div className="p-4 border-t border-border/80 flex items-center justify-between text-xs text-slate-400">
        <div>
          Showing{' '}
          <strong className="text-white">
            {filteredData.length === 0 ? 0 : (currentPage - 1) * itemsPerPage + 1}
          </strong>{' '}
          to{' '}
          <strong className="text-white">
            {Math.min(currentPage * itemsPerPage, filteredData.length)}
          </strong>{' '}
          of <strong className="text-white">{filteredData.length}</strong> activity logs
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-1.5 rounded-lg bg-background-elevated border border-border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-background-elevated/80 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="font-mono text-xs px-2">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="p-1.5 rounded-lg bg-background-elevated border border-border disabled:opacity-40 disabled:cursor-not-allowed hover:bg-background-elevated/80 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
