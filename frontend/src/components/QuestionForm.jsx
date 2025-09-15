import { useState } from 'react'

function QuestionForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('')
  const [answerType, setAnswerType] = useState('2_mark')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (question.trim() && !loading) {
      onSubmit(question.trim(), answerType)
      setQuestion('')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="answerType" className="block text-sm font-medium text-gray-700 mb-2">
          Answer Type
        </label>
        <select
          id="answerType"
          value={answerType}
          onChange={(e) => setAnswerType(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          disabled={loading}
        >
          <option value="2_mark">2-Mark Answer (4+ bullet points)</option>
          <option value="14_mark">14-Mark Answer (3-4 subheadings, 6-9 points each)</option>
        </select>
      </div>
      
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
