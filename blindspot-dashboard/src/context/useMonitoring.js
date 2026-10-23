import { useContext } from 'react';
import { MonitoringContext } from './monitoringContextDef';

export const useMonitoring = () => {
  const context = useContext(MonitoringContext);
  if (!context) {
    throw new Error('useMonitoring must be used within a MonitoringProvider');
  }
  return context;
};
