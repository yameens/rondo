'use client'

import { useState } from 'react'
import { BarChart3, CheckCircle, XCircle, Clock, Music, Download } from 'lucide-react'
import { AnalysisData, AccuracyMetrics } from '@/types'
import axios from 'axios'

interface AnalysisResultsProps {
  analysisData: AnalysisData
}

export default function AnalysisResults({ analysisData }: AnalysisResultsProps) {
  const [isExporting, setIsExporting] = useState(false)
  
  const metrics = analysisData.metrics
  if (!metrics) {
    return (
      <div className="text-center text-gray-500">
        No analysis results available
      </div>
    )
  }

  const handleExportCSV = async () => {
    try {
      setIsExporting(true)
      const response = await axios.get(`/api/analysis/${analysisData.id}/export/csv`)
      
      // Create and download file
      const blob = new Blob([response.data.content], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = response.data.filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Export failed:', error)
    } finally {
      setIsExporting(false)
    }
  }

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 0.9) return 'text-green-600'
    if (accuracy >= 0.7) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getAccuracyBgColor = (accuracy: number) => {
    if (accuracy >= 0.9) return 'bg-green-100'
    if (accuracy >= 0.7) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  return (
    <div className="space-y-6">
      {/* Overall Accuracy */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-gray-900">Overall Accuracy</h4>
          <BarChart3 className="h-5 w-5 text-blue-600" />
        </div>
        <div className="text-center">
          <div className={`text-4xl font-bold ${getAccuracyColor(metrics.overall_accuracy)}`}>
            {(metrics.overall_accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-sm text-gray-600 mt-1">Overall Performance Accuracy</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center mb-2">
            <CheckCircle className="h-4 w-4 text-green-600 mr-2" />
            <span className="text-sm font-medium text-gray-700">Correct Notes</span>
          </div>
          <div className="text-2xl font-bold text-green-600">{metrics.correct_notes}</div>
          <div className="text-xs text-gray-500">
            {metrics.total_score_notes > 0 
              ? `${((metrics.correct_notes / metrics.total_score_notes) * 100).toFixed(1)}% of total`
              : '0% of total'
            }
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center mb-2">
            <XCircle className="h-4 w-4 text-red-600 mr-2" />
            <span className="text-sm font-medium text-gray-700">Missed Notes</span>
          </div>
          <div className="text-2xl font-bold text-red-600">{metrics.missed_notes}</div>
          <div className="text-xs text-gray-500">
            {metrics.total_score_notes > 0 
              ? `${((metrics.missed_notes / metrics.total_score_notes) * 100).toFixed(1)}% of total`
              : '0% of total'
            }
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Music className="h-4 w-4 text-purple-600 mr-2" />
            <span className="text-sm font-medium text-gray-700">Extra Notes</span>
          </div>
          <div className="text-2xl font-bold text-purple-600">{metrics.extra_notes}</div>
          <div className="text-xs text-gray-500">
            {metrics.total_performance_notes > 0 
              ? `${((metrics.extra_notes / metrics.total_performance_notes) * 100).toFixed(1)}% of performance`
              : '0% of performance'
            }
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Clock className="h-4 w-4 text-orange-600 mr-2" />
            <span className="text-sm font-medium text-gray-700">Timing Errors</span>
          </div>
          <div className="text-2xl font-bold text-orange-600">{metrics.timing_errors}</div>
          <div className="text-xs text-gray-500">
            RMSE: {metrics.timing_rmse.toFixed(3)}s
          </div>
        </div>
      </div>

      {/* F1 Score */}
      <div className="bg-white border rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 mb-3">F1 Score Metrics</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-lg font-semibold text-blue-600">
              {(metrics.precision * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">Precision</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-green-600">
              {(metrics.recall * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">Recall</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-purple-600">
              {(metrics.f1_score * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">F1 Score</div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Summary Statistics</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Total Score Notes:</span>
            <span className="ml-2 font-medium">{metrics.total_score_notes}</span>
          </div>
          <div>
            <span className="text-gray-600">Total Performance Notes:</span>
            <span className="ml-2 font-medium">{metrics.total_performance_notes}</span>
          </div>
          <div>
            <span className="text-gray-600">Pitch Errors:</span>
            <span className="ml-2 font-medium">{metrics.pitch_errors}</span>
          </div>
          <div>
            <span className="text-gray-600">Timing RMSE:</span>
            <span className="ml-2 font-medium">{metrics.timing_rmse.toFixed(3)}s</span>
          </div>
        </div>
      </div>

      {/* Export Button */}
      <div className="flex justify-center">
        <button
          onClick={handleExportCSV}
          disabled={isExporting}
          className="flex items-center px-4 py-2 text-sm font-medium text-white bg-green-600 border border-transparent rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
        >
          <Download className="w-4 h-4 mr-2" />
          {isExporting ? 'Exporting...' : 'Export CSV Report'}
        </button>
      </div>
    </div>
  )
}
