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

// New verdict elements
const resultHero = document.getElementById('result-hero')
const heroIcon = document.getElementById('hero-icon')
const heroTitle = document.getElementById('hero-title')
const heroSubtitle = document.getElementById('hero-subtitle')
const metricConfidence = document.getElementById('metric-confidence')
const metricLatency = document.getElementById('metric-latency')

// Optional debug
const debugDetails = document.getElementById('debug-details')
const resultJson = document.getElementById('result-json')

const toastEl = document.getElementById('toast')
const toast = new bootstrap.Toast(toastEl)

// ===== GA helper (safe no-op if GA not loaded) =====
function gaEvent(name, params) {
  if (typeof window.gtag === 'function')
    window.gtag('event', name, params || {})
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
function resetUI() {
  selectedFile = null
  fileInput.value = ''
  previewWrap.classList.add('d-none')
  previewImg.src = ''
  fileNameEl.textContent = ''
  fileMetaEl.textContent = ''
  resultCard.classList.add('d-none')
  resultHero.classList.add('d-none')
  resultHero.classList.remove('good', 'bad')
  heroIcon.innerHTML = ''
  heroTitle.textContent = '—'
  heroSubtitle.textContent = '—'
  metricConfidence.textContent = '—'
  metricLatency.textContent = '—'
  if (resultJson) resultJson.textContent = ''
  debugDetails?.classList.add('d-none')
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

  gaEvent('select_image', {
    via: via || 'unknown',
    file_type: file.type || 'n/a',
    file_size_kb: Math.round(file.size / 1024),
  })
}

// ===== Events =====
dropzone.addEventListener('click', () => fileInput.click())
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click()
})
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
fileInput.addEventListener('change', (e) =>
  fileChosen(e.target.files[0], 'picker')
)
window.addEventListener('paste', (e) => {
  const item = [...e.clipboardData.items].find((i) =>
    i.type.startsWith('image/')
  )
  if (item) fileChosen(item.getAsFile(), 'paste')
})
clearBtn.addEventListener('click', resetUI)

// ===== Predict =====
predictBtn.addEventListener('click', async () => {
  const file = selectedFile
  if (!file) {
    showToast('Choose an image first.')
    return
  }

  predictBtn.disabled = true
  spinner.classList.remove('d-none')
  resultCard.classList.remove('d-none')
  resultHero.classList.add('d-none')
  debugDetails?.classList.add('d-none')

  gaEvent('predict_started')

  const t0 = performance.now()
  try {
    const formData = new FormData()
    formData.append('file', file)

    const resp = await fetch('/predict', { method: 'POST', body: formData })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()

    // --- Optional: keep raw for debugging, but hidden ---
    if (resultJson) resultJson.textContent = JSON.stringify(data, null, 2)

    // --- Interpret common shapes ---
    // label: "Tumor" / "No Tumor" / or subtype
    let labelText = 'Prediction'
    if ('prediction' in data) {
      labelText =
        data.prediction === 1 ||
        String(data.prediction).toLowerCase() === 'tumor'
          ? 'Tumor'
          : 'No Tumor'
    } else if ('label' in data) {
      labelText = String(data.label)
    } else if ('class' in data) {
      labelText = String(data.class)
    }

    // confidence
    let score = null
    if (typeof data.score === 'number') score = data.score
    else if (typeof data.confidence === 'number') score = data.confidence
    else if (typeof data.probability === 'number') score = data.probability

    const pct =
      score !== null && !Number.isNaN(score)
        ? Math.round(Math.max(0, Math.min(1, score)) * 100)
        : null

    // latency (prefer server time if provided)
    const clientMs = Math.round(performance.now() - t0)
    const serverMs =
      typeof data.inference_ms === 'number' ? data.inference_ms : null

    // --- Decide verdict ---
    const labelLower = labelText.trim().toLowerCase()
    const isTumor =
      (labelLower.includes('tumor') && !labelLower.includes('no tumor')) ||
      [
        'glioma',
        'meningioma',
        'pituitary',
        'pituitary_tumor',
        'gbm',
        'lgg',
      ].some((k) => labelLower.includes(k))

    // --- Render verdict hero ---
    resultHero.classList.remove('good', 'bad')
    resultHero.classList.add(isTumor ? 'bad' : 'good')

    // Icon SVGs
    const checkSvg = `
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="11" stroke="currentColor" opacity=".25"></circle>
        <path d="M7 12.5l3.2 3.2L17 9.8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`
    const alertSvg = `
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="11" stroke="currentColor" opacity=".25"></circle>
        <path d="M12 7v6M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`

    heroIcon.innerHTML = isTumor ? alertSvg : checkSvg
    heroTitle.textContent = isTumor ? 'Tumor detected' : 'No tumor detected'

    // Subtitle: show subtype if present, else neutral text
    const subtype = ['glioma', 'meningioma', 'pituitary', 'gbm', 'lgg'].find(
      (k) => labelLower.includes(k)
    )
    heroSubtitle.textContent = subtype
      ? `Subtype: ${subtype.charAt(0).toUpperCase() + subtype.slice(1)}`
      : labelText && labelText !== 'Prediction'
      ? `Model label: ${labelText}`
      : 'Model result'

    metricConfidence.textContent = pct !== null ? `${pct}%` : '—'
    metricLatency.textContent = serverMs
      ? `${serverMs} ms (server)`
      : `${clientMs} ms (round-trip)`

    resultHero.classList.remove('d-none')

    gaEvent('predict_success', {
      label: labelText,
      confidence_pct: pct ?? undefined,
      roundtrip_ms: clientMs,
    })
  } catch (err) {
    showToast('Prediction failed: ' + err.message)
    resultHero.classList.add('d-none')
    gaEvent('predict_error', { message: String(err.message || err) })
  } finally {
    spinner.classList.add('d-none')
    predictBtn.disabled = false
  }
})
