function AnswerDisplay({ answer }) {
  const formatExplanation = (text) => {
    if (!text) return null;
    
    // Split by lines and process each line
    const lines = text.split('\n').filter(line => line.trim());
    const elements = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Check if it's a subheading (starts with number like "2.1", "2.2", etc.)
      if (/^\d+\.\d+\s+/.test(line)) {
        elements.push(
          <h4 key={i} className="text-md font-semibold text-gray-800 mt-4 mb-2 first:mt-0">
            {line}
          </h4>
        );
      }
      // Check if it's a bullet point (starts with "-")
      else if (line.startsWith('-')) {
        elements.push(
          <div key={i} className="ml-4 mb-1">
            <span className="text-gray-700">• {line.substring(1).trim()}</span>
          </div>
        );
      }
      // Regular paragraph
      else if (line) {
        elements.push(
          <p key={i} className="mb-2 text-gray-700">
            {line}
          </p>
        );
      }
    }
    
    return elements.length > 0 ? elements : <p className="text-gray-700">{text}</p>;
  };

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
          <div className="text-gray-700 leading-relaxed">
            {formatExplanation(answer.explanation)}
          </div>
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
