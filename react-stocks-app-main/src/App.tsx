import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import MainLayout from "@/layouts/main-layout";
import { Toaster } from "@/components/ui/toaster";
import LoadingIndicator from "@/components/loading-indicator";
import PredictionPage from "./pages/prediction-page";

// lazy loading the pages to improve performance
const DashboardPage = React.lazy(() => import("@/pages/dashboard-page"));
const StockDetailsPage = React.lazy(() => import("@/pages/stock-details-page"));
const SettingsPage = React.lazy(() => import("@/pages/settings-page"));
const PortfolioPage = React.lazy(() => import("@/pages/portfolio-page"));
const NewsPage = React.lazy(() => import("@/pages/news-page"));

export default function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route
            path="/"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <DashboardPage />
              </React.Suspense>
            }
          />
          <Route
            path="/stock/:symbol"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <StockDetailsPage />
              </React.Suspense>
            }
          />
          <Route
            path="/settings"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <SettingsPage />
              </React.Suspense>
            }
          />
          <Route
            path="/portfolio"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <PortfolioPage />
              </React.Suspense>
            }
          />
          <Route
            path="/news"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <NewsPage />
              </React.Suspense>
            }
          />
          <Route
            path="/prediction"
            element={
              <React.Suspense fallback={<LoadingIndicator />}>
                <PredictionPage />
              </React.Suspense>
            }
          />
        </Routes>

        <Toaster />
      </MainLayout>
    </BrowserRouter>
  );
}
