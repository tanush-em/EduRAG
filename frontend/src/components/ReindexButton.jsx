function ReindexButton({ onReindex, loading }) {
  return (
    <button
      onClick={onReindex}
      disabled={loading}
      className="px-4 py-2 bg-gray-600 text-white text-sm font-medium rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      title="Reload and reindex all notes from the notes directory"
    >
      {loading ? 'Reindexing...' : 'Reindex Notes'}
    </button>
  )
}

export default ReindexButton
