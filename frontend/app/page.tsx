'use client'

import { useState } from 'react'
import { Upload, Music, FileAudio, BarChart3, Download } from 'lucide-react'
import FileUpload from '@/components/FileUpload'
import AnalysisResults from '@/components/AnalysisResults'
import ScoreViewer from '@/components/ScoreViewer'
import { AnalysisData } from '@/types'

export default function Home() {
  const [analysisId, setAnalysisId] = useState<string | null>(null)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleAnalysisComplete = (data: AnalysisData) => {
    setAnalysisData(data)
    setIsLoading(false)
  }

  const handleUploadStart = () => {
    setIsLoading(true)
    setAnalysisData(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Music className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-3xl font-bold text-gray-900">Rondo</h1>
              <span className="ml-2 text-sm text-gray-500">Classical Performance Analysis</span>
            </div>
            <div className="flex items-center space-x-4">
              <BarChart3 className="h-5 w-5 text-gray-400" />
              <span className="text-sm text-gray-500">v1.0</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!analysisData ? (
          <div className="space-y-8">
            {/* Upload Section */}
            <div className="bg-white rounded-lg shadow-sm border p-8">
              <div className="text-center mb-8">
                <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
                  <Upload className="h-6 w-6 text-blue-600" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                  Upload Your Performance
                </h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Upload a score (MusicXML, PDF) and audio recording (WAV, MP3) to analyze 
                  where your performance diverges from the written score.
                </p>
              </div>

              <FileUpload 
                onUploadStart={handleUploadStart}
                onAnalysisComplete={handleAnalysisComplete}
                isLoading={isLoading}
              />
            </div>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex items-center mb-4">
                  <FileAudio className="h-8 w-8 text-green-600 mr-3" />
                  <h3 className="text-lg font-semibold text-gray-900">Audio Transcription</h3>
                </div>
                <p className="text-gray-600">
                  Advanced AI transcription using Basic Pitch to convert your performance 
                  into precise note events with timing and velocity data.
                </p>
              </div>

              <div className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex items-center mb-4">
                  <BarChart3 className="h-8 w-8 text-purple-600 mr-3" />
                  <h3 className="text-lg font-semibold text-gray-900">Score Alignment</h3>
                </div>
                <p className="text-gray-600">
                  Intelligent alignment algorithms match your performance to the score, 
                  identifying timing deviations, missed notes, and extra notes.
                </p>
              </div>

              <div className="bg-white rounded-lg shadow-sm border p-6">
                <div className="flex items-center mb-4">
                  <Download className="h-8 w-8 text-orange-600 mr-3" />
                  <h3 className="text-lg font-semibold text-gray-900">Detailed Reports</h3>
                </div>
                <p className="text-gray-600">
                  Export comprehensive analysis reports including CSV data, 
                  annotated MusicXML, and visual accuracy heatmaps.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Results Header */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900">Analysis Results</h2>
                  <p className="text-gray-600">Performance accuracy analysis completed</p>
                </div>
                <button
                  onClick={() => {
                    setAnalysisData(null)
                    setAnalysisId(null)
                  }}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
                >
                  New Analysis
                </button>
              </div>
            </div>

            {/* Results Content */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Score Viewer */}
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Score Visualization</h3>
                <ScoreViewer analysisData={analysisData} />
              </div>

              {/* Analysis Results */}
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Accuracy Analysis</h3>
                <AnalysisResults analysisData={analysisData} />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-500">
            <p>&copy; 2024 Rondo. Classical Performance Analysis Tool.</p>
            <p className="mt-2 text-sm">
              Built with Next.js, FastAPI, and Basic Pitch for accurate musical analysis.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
