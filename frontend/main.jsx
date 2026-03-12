import { createRoot } from 'react-dom/client';
import App from './app.js';
import ErrorBoundary from './ErrorBoundary.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
