import { useState } from 'react'

function QuestionForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (question.trim() && !loading) {
      onSubmit(question.trim())
      setQuestion('')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
          Your Question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Type your question here... (e.g., What is photosynthesis?)"
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
          rows={3}
          disabled={loading}
        />
      </div>
      
      <div className="flex justify-end">
        <button
          type="submit"
          disabled={!question.trim() || loading}
          className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Asking...' : 'Ask Question'}
        </button>
      </div>
    </form>
  )
}

export default QuestionForm
