import { useState } from 'react'
import QuestionForm from './components/QuestionForm'
import AnswerDisplay from './components/AnswerDisplay'
import ReindexButton from './components/ReindexButton'

function App() {
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleQuestionSubmit = async (question) => {
    setLoading(true)
    setError(null)
    setAnswer(null)

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setAnswer(data)
    } catch (err) {
      setError(err.message)
      console.error('Error asking question:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReindex = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/reindex', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      alert(`Success: ${data.message}`)
    } catch (err) {
      setError(err.message)
      console.error('Error reindexing:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            EduRAG
          </h1>
          <p className="text-lg text-gray-600">
            Ask questions about your study notes and get structured answers
          </p>
        </header>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-800">
              Ask a Question
            </h2>
            <ReindexButton onReindex={handleReindex} loading={loading} />
          </div>
          
          <QuestionForm onSubmit={handleQuestionSubmit} loading={loading} />
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Error
                </h3>
                <div className="mt-2 text-sm text-red-700">
                  {error}
                </div>
              </div>
            </div>
          </div>
        )}

        {answer && (
          <AnswerDisplay answer={answer} />
        )}

        {loading && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Processing your question...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App