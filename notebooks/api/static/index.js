'use strict'

// ===== State =====
let selectedFile = null

// ===== Elements =====
const dropzone = document.getElementById('dropzone')
const fileInput = document.getElementById('file-input')
const previewWrap = document.getElementById('preview-wrap')
const previewImg = document.getElementById('preview-img')
const fileNameEl = document.getElementById('file-name')
const fileMetaEl = document.getElementById('file-meta')
const predictBtn = document.getElementById('predict-btn')
const spinner = document.getElementById('spinner')
const clearBtn = document.getElementById('clear-btn')
const resultCard = document.getElementById('result-card')
const resultJson = document.getElementById('result-json')
const badge = document.getElementById('badge')
const confidenceWrap = document.getElementById('confidence-wrap')
const confidenceBar = document.getElementById('confidence')
const timingEl = document.getElementById('timing')
const toastEl = document.getElementById('toast')
const toast = new bootstrap.Toast(toastEl)

// ===== GA helper (no-op until GA is loaded after consent) =====
function gaEvent(name, params) {
  if (typeof window.gtag === 'function') {
    window.gtag('event', name, params || {})
  }
}

// ===== Config =====
const MAX_MB = 10
const ACCEPTED = ['image/jpeg', 'image/png']

// ===== Helpers =====
function showToast(msg) {
  document.getElementById('toast-msg').textContent = msg
  toast.show()
}

function humanSize(bytes) {
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const sizes = ['B', 'KB', 'MB', 'GB']
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + sizes[i]
}

function setBadge(label, kind) {
  badge.className = 'badge rounded-pill ' + (kind || 'text-bg-secondary')
  badge.textContent = label
}

function resetUI() {
  selectedFile = null
  fileInput.value = ''
  previewWrap.classList.add('d-none')
  previewImg.src = ''
  fileNameEl.textContent = ''
  fileMetaEl.textContent = ''
  predictBtn.disabled = true
  resultCard.classList.add('d-none')
  confidenceWrap.classList.add('d-none')
  confidenceBar.style.width = '0%'
  confidenceBar.textContent = '0%'
  timingEl.textContent = ''
}

// Validate + preview + enable button
function fileChosen(file, via) {
  if (!file) return

  if (!file.type || !ACCEPTED.includes(file.type)) {
    showToast('Unsupported file type. Please use JPG or PNG.')
    return
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    showToast(`File too large. Max ${MAX_MB} MB.`)
    return
  }

  selectedFile = file

  const reader = new FileReader()
  reader.onload = (e) => {
    previewImg.src = e.target.result
  }
  reader.readAsDataURL(file)

  fileNameEl.textContent = file.name
  fileMetaEl.textContent = `${file.type || 'image'} • ${humanSize(file.size)}`
  previewWrap.classList.remove('d-none')
  predictBtn.disabled = false

  // Analytics (fires only after consent/GA load)
  gaEvent('select_image', {
    via: via || 'unknown',
    file_type: file.type || 'n/a',
    file_size_kb: Math.round(file.size / 1024),
  })
}

// ===== Events =====
// Open file dialog
dropzone.addEventListener('click', () => fileInput.click())
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click()
})

// Drag&Drop styling + file selection
;['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropzone.classList.add('dragover')
  })
})
;['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropzone.classList.remove('dragover')
  })
})
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0]
  fileChosen(file, 'drag_drop')
})

// File picker
fileInput.addEventListener('change', (e) =>
  fileChosen(e.target.files[0], 'picker')
)

// Paste image from clipboard
window.addEventListener('paste', (e) => {
  const item = [...e.clipboardData.items].find((i) =>
    i.type.startsWith('image/')
  )
  if (item) {
    const file = item.getAsFile()
    fileChosen(file, 'paste')
  }
})

// Clear UI
clearBtn.addEventListener('click', resetUI)

// Predict
predictBtn.addEventListener('click', async () => {
  const file = selectedFile
  if (!file) {
    showToast('Choose an image first.')
    return
  }

  predictBtn.disabled = true
  spinner.classList.remove('d-none')
  setBadge('Predicting…', 'text-bg-warning')
  resultCard.classList.remove('d-none')
  resultJson.textContent = ''
  confidenceWrap.classList.add('d-none')
  timingEl.textContent = ''

  // Analytics: prediction started
  gaEvent('predict_started')

  const t0 = performance.now()
  try {
    const formData = new FormData()
    formData.append('file', file)

    const resp = await fetch('/predict', { method: 'POST', body: formData })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

    const data = await resp.json()
    resultJson.textContent = JSON.stringify(data, null, 2)

    // Interpret common shapes
    const label =
      'prediction' in data
        ? data.prediction === 1
          ? 'Tumor'
          : 'No Tumor'
        : 'label' in data
        ? String(data.label).toLowerCase().includes('tumor')
          ? 'Tumor'
          : 'No Tumor'
        : 'Prediction'

    const score =
      typeof data.score === 'number'
        ? data.score
        : typeof data.confidence === 'number'
        ? data.confidence
        : typeof data.probability === 'number'
        ? data.probability
        : null

    if (label === 'Tumor') setBadge('Tumor', 'text-bg-danger')
    else if (label === 'No Tumor') setBadge('No Tumor', 'text-bg-success')
    else setBadge(label, 'text-bg-info')

    if (score !== null && !Number.isNaN(score)) {
      const pct = Math.round(Math.max(0, Math.min(1, score)) * 100)
      confidenceBar.style.width = pct + '%'
      confidenceBar.textContent = pct + '%'
      confidenceWrap.classList.remove('d-none')
    }

    const clientMs = Math.round(performance.now() - t0)
    timingEl.textContent = `Round-trip: ${clientMs} ms`

    // Analytics: prediction succeeded
    gaEvent('predict_success', {
      label,
      confidence_pct:
        typeof score === 'number'
          ? Math.round(Math.max(0, Math.min(1, score)) * 100)
          : undefined,
      roundtrip_ms: clientMs,
    })
  } catch (err) {
    setBadge('Error', 'text-bg-danger')
    showToast('Prediction failed: ' + err.message)

    // Analytics: prediction error
    gaEvent('predict_error', { message: String(err.message || err) })
  } finally {
    spinner.classList.add('d-none')
    predictBtn.disabled = false
  }
})
