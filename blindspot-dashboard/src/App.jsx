import React from 'react';
import { MonitoringProvider, useMonitoring } from './context/MonitoringContext';
import { Layout } from './components/layout/Layout';
import { OverviewPage } from './pages/OverviewPage';
import { LiveActivityPage } from './pages/LiveActivityPage';
import { ScreenContextPage } from './pages/ScreenContextPage';
import { ActivityHistoryPage } from './pages/ActivityHistoryPage';
import { ActivityIntelligencePage } from './pages/ActivityIntelligencePage';
import { SettingsPage } from './pages/SettingsPage';

const AppContent = () => {
  const { currentPage } = useMonitoring();

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <OverviewPage />;
      case 'live':
        return <LiveActivityPage />;
      case 'context':
        return <ScreenContextPage />;
      case 'history':
        return <ActivityHistoryPage />;
      case 'intelligence':
        return <ActivityIntelligencePage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <OverviewPage />;
    }
  };

  return <Layout>{renderPage()}</Layout>;
};

export const App = () => {
  return (
    <MonitoringProvider>
      <AppContent />
    </MonitoringProvider>
  );
};

export default App;
