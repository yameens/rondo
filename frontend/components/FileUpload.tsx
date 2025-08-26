'use client'

import { useState, useRef } from 'react'
import { Upload, FileAudio, FileText, Loader2, AlertCircle } from 'lucide-react'
import axios from 'axios'
import { AnalysisData, UploadResponse } from '@/types'

interface FileUploadProps {
  onUploadStart: () => void
  onAnalysisComplete: (data: AnalysisData) => void
  isLoading: boolean
}

export default function FileUpload({ onUploadStart, onAnalysisComplete, isLoading }: FileUploadProps) {
  const [scoreFile, setScoreFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [onsetTolerance, setOnsetTolerance] = useState(500)
  const [pitchTolerance, setPitchTolerance] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const [analysisId, setAnalysisId] = useState<string | null>(null)
  
  const scoreInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleScoreFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setScoreFile(file)
      setError(null)
    }
  }

  const handleAudioFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!scoreFile || !audioFile) {
      setError('Please select both score and audio files')
      return
    }

    try {
      onUploadStart()
      setError(null)

      const formData = new FormData()
      formData.append('score_file', scoreFile)
      formData.append('audio_file', audioFile)
      formData.append('onset_tolerance_ms', onsetTolerance.toString())
      formData.append('pitch_tolerance', pitchTolerance.toString())

      const response = await axios.post<UploadResponse>('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setAnalysisId(response.data.id)
      
      // Poll for results
      pollAnalysisResults(response.data.id)
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.')
      onUploadStart() // Reset loading state
    }
  }

  const pollAnalysisResults = async (id: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`/api/analysis/${id}`)
        const data = response.data

        if (data.status === 'completed') {
          clearInterval(pollInterval)
          onAnalysisComplete(data)
        } else if (data.status === 'failed') {
          clearInterval(pollInterval)
          setError(data.error_message || 'Analysis failed')
          onUploadStart() // Reset loading state
        }
        // Continue polling if status is 'pending' or 'processing'
      } catch (err) {
        clearInterval(pollInterval)
        setError('Failed to check analysis status')
        onUploadStart() // Reset loading state
      }
    }, 2000) // Poll every 2 seconds
  }

  const resetForm = () => {
    setScoreFile(null)
    setAudioFile(null)
    setError(null)
    setAnalysisId(null)
    if (scoreInputRef.current) scoreInputRef.current.value = ''
    if (audioInputRef.current) audioInputRef.current.value = ''
  }

  return (
    <div className="max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Score File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Score File (MusicXML, PDF)
          </label>
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                {scoreFile ? (
                  <>
                    <FileText className="w-8 h-8 mb-2 text-blue-600" />
                    <p className="text-sm text-gray-600">{scoreFile.name}</p>
                  </>
                ) : (
                  <>
                    <FileText className="w-8 h-8 mb-2 text-gray-400" />
                    <p className="text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">MusicXML, PDF (max 100MB)</p>
                  </>
                )}
              </div>
              <input
                ref={scoreInputRef}
                type="file"
                className="hidden"
                accept=".xml,.musicxml,.mxl,.mei,.pdf"
                onChange={handleScoreFileChange}
                disabled={isLoading}
              />
            </label>
          </div>
        </div>

        {/* Audio File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Audio Recording (WAV, MP3)
          </label>
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                {audioFile ? (
                  <>
                    <FileAudio className="w-8 h-8 mb-2 text-green-600" />
                    <p className="text-sm text-gray-600">{audioFile.name}</p>
                  </>
                ) : (
                  <>
                    <FileAudio className="w-8 h-8 mb-2 text-gray-400" />
                    <p className="text-sm text-gray-500">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-gray-500">WAV, MP3 (max 100MB)</p>
                  </>
                )}
              </div>
              <input
                ref={audioInputRef}
                type="file"
                className="hidden"
                accept=".wav,.mp3,.flac"
                onChange={handleAudioFileChange}
                disabled={isLoading}
              />
            </label>
          </div>
        </div>

        {/* Tolerance Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Onset Tolerance (ms)
            </label>
            <input
              type="range"
              min="50"
              max="1000"
              value={onsetTolerance}
              onChange={(e) => setOnsetTolerance(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              disabled={isLoading}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>50ms</span>
              <span className="font-medium">{onsetTolerance}ms</span>
              <span>1000ms</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Pitch Tolerance (semitones)
            </label>
            <input
              type="range"
              min="0"
              max="12"
              value={pitchTolerance}
              onChange={(e) => setPitchTolerance(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              disabled={isLoading}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0</span>
              <span className="font-medium">{pitchTolerance}</span>
              <span>12</span>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="flex items-center p-4 text-sm text-red-800 border border-red-200 rounded-lg bg-red-50">
            <AlertCircle className="w-4 h-4 mr-2" />
            {error}
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-center">
          <button
            type="submit"
            disabled={isLoading || !scoreFile || !audioFile}
            className="flex items-center px-6 py-3 text-base font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5 mr-2" />
                Start Analysis
              </>
            )}
          </button>
        </div>

        {/* Reset Button */}
        {!isLoading && (scoreFile || audioFile) && (
          <div className="flex justify-center">
            <button
              type="button"
              onClick={resetForm}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Reset Form
            </button>
          </div>
        )}
      </form>
    </div>
  )
}
