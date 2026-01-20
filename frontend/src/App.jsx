
import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  // Search State
  const [speciesList, setSpeciesList] = useState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [searchResult, setSearchResult] = useState(null)

  const fileInputRef = useRef(null)

  useEffect(() => {
    fetch('http://localhost:8000/species')
      .then(res => res.json())
      .then(data => setSpeciesList(data.species))
      .catch(err => console.error("Failed to fetch species list", err))
  }, [])

  const handleSearchChange = (e) => {
    const query = e.target.value
    setSearchQuery(query)

    if (query.length > 0) {
      const filtered = speciesList.filter(s =>
        s.toLowerCase().includes(query.toLowerCase())
      )
      setSuggestions(filtered.slice(0, 5)) // Limit suggestions
    } else {
      setSuggestions([])
    }
  }

  const fetchButterflyImage = async (speciesName) => {
    try {
      // 1. Try a search query to find the best matching page (most robust)
      const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search&gsrsearch=${encodeURIComponent(speciesName + " butterfly")}&gsrlimit=1&prop=pageimages&pithumbsize=800&origin=*`
      const searchRes = await fetch(searchUrl)
      const searchData = await searchRes.json()

      if (searchData.query && searchData.query.pages) {
        const pages = searchData.query.pages
        const pageId = Object.keys(pages)[0]
        if (pages[pageId].thumbnail) {
          return pages[pageId].thumbnail.source
        }
      }

      // 2. Fallback to direct title search with redirects (handles case sensitivity)
      const directUrl = `https://en.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(speciesName)}&redirects=1&prop=pageimages&pithumbsize=800&origin=*`
      const directRes = await fetch(directUrl)
      const directData = await directRes.json()

      if (directData.query && directData.query.pages) {
        const pages = directData.query.pages
        const pageId = Object.keys(pages)[0]
        if (pageId !== "-1" && pages[pageId].thumbnail) {
          return pages[pageId].thumbnail.source
        }
      }
    } catch (err) {
      console.error("Wikipedia fetch error", err)
    }
    return null
  }

  const handleSelectSpecies = async (species) => {
    setSearchQuery(species)
    setSuggestions([])
    setLoading(true)
    setFile(null)
    setPreview(null)
    setResult(null)

    // Fetch image
    const imageUrl = await fetchButterflyImage(species)

    setSearchResult({
      name: species,
      image: imageUrl
    })
    setLoading(false)
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
    setSearchResult(null)
    uploadFile(selectedFile)
  }

  const uploadFile = async (selectedFile) => {
    setLoading(true)
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error(err)
      setError('Failed to classify image. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const resetApp = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <div className="main-content">
        <h1 className="main-title">🦋 Butterfly Species Classifier</h1>
        <p className="description">Upload a butterfly image and I'll tell you its species!</p>

        {/* Search Section (Keeping it but styling it cleaner) */}
        <div className="search-section">
          <p className="section-label">Choose an image...</p>
          <div className="search-wrapper">
            <input
              type="text"
              className="search-bar"
              placeholder="Search butterfly name..."
              value={searchQuery}
              onChange={handleSearchChange}
            />
            {suggestions.length > 0 && (
              <ul className="suggestions">
                {suggestions.map((s, index) => (
                  <li key={index} onClick={() => handleSelectSpecies(s)}>
                    {s}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* File Uploader */}
        <div
          className={`upload-container ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="input-file"
            accept=".jpg,.jpeg,.png"
            onChange={handleChange}
          />
          <div className="upload-box">
            <div className="upload-info">
              <span className="cloud-icon">☁️</span>
              <div className="upload-text">
                <p className="primary-text">Drag and drop file here</p>
                <p className="secondary-text">Limit 200MB per file • JPG, JPEG, PNG</p>
              </div>
            </div>
            <button className="browse-btn" onClick={() => fileInputRef.current.click()}>
              Browse files
            </button>
          </div>
        </div>

        {/* File List Item / Progress */}
        {file && (
          <div className="file-item">
            <div className="file-info">
              <span className="file-icon">📄</span>
              <div className="file-details">
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            </div>
            <button className="close-file" onClick={resetApp}>×</button>
          </div>
        )}

        {/* Results / Preview Section */}
        {(preview || searchResult) && (
          <div className="results-wrapper">
            {loading && <div className="spinner"></div>}

            {(preview || (searchResult && searchResult.image)) && (
              <div className="image-preview-card">
                <img
                  src={preview || searchResult.image}
                  alt="Butterfly"
                  className="butterfly-image"
                />
              </div>
            )}

            {result && (
              <div className="classification-result">
                <div className="success-banner">
                  ✅ Predicted: <strong>{result.label}</strong>
                </div>
                <div className="confidence-label">
                  🔍 Confidence: {(result.confidence * 100).toFixed(2)}%
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            )}

            {searchResult && !file && (
              <div className="search-info">
                <p>Showing info for: <strong>{searchResult.name}</strong></p>
                {!searchResult.image && <p className="no-image">No image found for this species.</p>}
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
