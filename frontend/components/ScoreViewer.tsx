'use client'

import { useEffect, useRef, useState } from 'react'
import { AnalysisData } from '@/types'

interface ScoreViewerProps {
  analysisData: AnalysisData
}

export default function ScoreViewer({ analysisData }: ScoreViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const appRef = useRef<any>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load Verovio dynamically
  useEffect(() => {
    let cancelled = false
    
    const loadVerovio = async () => {
      try {
        // Load Verovio script dynamically
        const script = document.createElement('script')
        script.src = 'https://www.verovio.org/javascript/app/verovio-app.js'
        script.async = true
        
        script.onload = () => {
          if (cancelled) return
          
          // Initialize Verovio App
          if (containerRef.current && (window as any).Verovio) {
            // @ts-ignore
            appRef.current = new (window as any).Verovio.App(containerRef.current, {
              defaultZoom: 50,
              defaultPageWidth: 1200,
              defaultPageHeight: 1600,
              adjustPageHeight: true,
              adjustPageWidth: true,
              border: 50,
              format: 'xml',
              font: 'Leipzig',
              scale: 50,
              scaleToPageSize: false,
              xmlIdSeed: 1
            })
            setReady(true)
          }
        }
        
        script.onerror = () => {
          console.error('Failed to load Verovio script')
          setError('Failed to load score viewer')
        }
        
        document.head.appendChild(script)
        
        return () => {
          if (script.parentNode) {
            script.parentNode.removeChild(script)
          }
        }
      } catch (err) {
        console.error('Failed to load Verovio:', err)
        setError('Failed to load score viewer')
      }
    }

    loadVerovio()
    
    return () => {
      cancelled = true
    }
  }, [])

  // Load sample score when ready
  useEffect(() => {
    if (!ready || !appRef.current) return

    // For MVP, load a sample score
    // In production, this would load the actual score from the analysis
    const loadSampleScore = async () => {
      try {
        const response = await fetch("https://www.verovio.org/editor/brahms.mei")
        const scoreText = await response.text()
        appRef.current.loadData(scoreText, { from: "mei" })
      } catch (err) {
        console.error('Failed to load sample score:', err)
        setError('Failed to load score')
      }
    }

    loadSampleScore()
  }, [ready])

  // Apply analysis overlays when data is available
  useEffect(() => {
    if (!ready || !appRef.current || !analysisData.results) return

    // Apply visual overlays based on analysis results
    applyAnalysisOverlays(analysisData.results)
  }, [ready, analysisData.results])

  const applyAnalysisOverlays = (results: any[]) => {
    if (!appRef.current) return

    // This is a placeholder for applying visual overlays
    // In a full implementation, this would:
    // 1. Map analysis results to score elements
    // 2. Apply color overlays for different error types
    // 3. Add click handlers for detailed information
    
    console.log('Applying analysis overlays to score:', results.length, 'results')
    
    // Example overlay application (would need proper implementation)
    results.forEach((result, index) => {
      if (result.accuracy_type === 'missed') {
        // Apply red overlay for missed notes
        console.log(`Missed note at index ${index}`)
      } else if (result.accuracy_type === 'extra') {
        // Apply purple overlay for extra notes
        console.log(`Extra note at index ${index}`)
      } else if (result.accuracy_type === 'timing_error') {
        // Apply orange overlay for timing errors
        console.log(`Timing error at index ${index}`)
      }
    })
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!appRef.current || !e.target.files?.[0]) return
    
    const file = e.target.files[0]
    try {
      const text = await file.text()
      const ext = file.name.toLowerCase().endsWith(".xml") || file.name.toLowerCase().endsWith(".musicxml")
        ? "musicxml"
        : "mei"
      appRef.current.loadData(text, { from: ext })
    } catch (err) {
      console.error('Failed to load file:', err)
      setError('Failed to load score file')
    }
  }

  if (error) {
    return (
      <div className="text-center text-red-600 p-4">
        <p>{error}</p>
        <button 
          onClick={() => setError(null)}
          className="mt-2 text-sm text-blue-600 hover:text-blue-800"
        >
          Try again
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <input 
            type="file" 
            accept=".mei,.xml,.musicxml" 
            onChange={handleFileUpload}
            className="text-sm"
          />
          {!ready && <span className="text-sm text-gray-500">Loading Verovio…</span>}
        </div>
        
        {/* Legend */}
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded mr-1"></div>
            <span>Missed</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded mr-1"></div>
            <span>Extra</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-orange-500 rounded mr-1"></div>
            <span>Timing</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded mr-1"></div>
            <span>Correct</span>
          </div>
        </div>
      </div>

      {/* Score Container */}
      <div 
        ref={containerRef} 
        className="border border-gray-200 rounded-lg bg-white overflow-auto"
        style={{ minHeight: '600px', maxHeight: '800px' }}
      />
      
      {/* Analysis Summary */}
      {analysisData.results && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Score Analysis Summary</h4>
          <div className="grid grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-gray-600">Total Events:</span>
              <span className="ml-1 font-medium">{analysisData.results.length}</span>
            </div>
            <div>
              <span className="text-gray-600">Correct:</span>
              <span className="ml-1 font-medium text-green-600">
                {analysisData.results.filter(r => r.accuracy_type === 'correct').length}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Errors:</span>
              <span className="ml-1 font-medium text-red-600">
                {analysisData.results.filter(r => r.accuracy_type !== 'correct').length}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Accuracy:</span>
              <span className="ml-1 font-medium">
                {analysisData.metrics ? `${(analysisData.metrics.overall_accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
