function AnswerDisplay({ answer }) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">Answer</h2>
      
      <div className="space-y-6">
        {/* Direct Answer */}
        <div className="border-l-4 border-blue-500 pl-4">
          <h3 className="text-lg font-medium text-gray-800 mb-2">
            Direct Answer
          </h3>
          <p className="text-gray-700 leading-relaxed">
            {answer.direct_answer}
          </p>
        </div>

        {/* Explanation */}
        <div className="border-l-4 border-green-500 pl-4">
          <h3 className="text-lg font-medium text-gray-800 mb-2">
            Explanation
          </h3>
          <p className="text-gray-700 leading-relaxed">
            {answer.explanation}
          </p>
        </div>

        {/* Summary */}
        <div className="border-l-4 border-purple-500 pl-4">
          <h3 className="text-lg font-medium text-gray-800 mb-2">
            Summary
          </h3>
          <p className="text-gray-700 leading-relaxed">
            {answer.summary}
          </p>
        </div>
      </div>
    </div>
  )
}

export default AnswerDisplay
