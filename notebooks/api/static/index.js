'use strict'

/**
 * Full front-end logic for the MRI predictor UI.
 * - Robustly parses many response shapes, including YOURS:
 *   {
 *     "label_id": 1,
 *     "label_name": "tumor",
 *     "probability_tumor": 0.9479,
 *     "threshold": 0.05
 *   }
 * - Renders a clean verdict card (expects the "verdict hero" HTML I shared earlier).
 * - GA event calls are safe no-ops if GA isn't loaded.
 */

/* ========================== State & Elements ========================== */
let selectedFile = null

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
const resultHero = document.getElementById('result-hero')
const heroIcon = document.getElementById('hero-icon')
const heroTitle = document.getElementById('hero-title')
const heroSubtitle = document.getElementById('hero-subtitle')
const metricConfidence = document.getElementById('metric-confidence')
const metricLatency = document.getElementById('metric-latency')

// Optional debug section (if present in your HTML)
const debugDetails = document.getElementById('debug-details')
const resultJson = document.getElementById('result-json')

// Toast
const toastEl = document.getElementById('toast')
const toast = new bootstrap.Toast(toastEl)

/* ============================== Config =============================== */
const MAX_MB = 10
const ACCEPTED = ['image/jpeg', 'image/png']

/* ============================ Analytics ============================= */
function gaEvent(name, params) {
  if (typeof window.gtag === 'function')
    window.gtag('event', name, params || {})
}

/* ============================= Helpers ============================== */
function showToast(msg) {
  document.getElementById('toast-msg').textContent = msg
  toast.show()
}
function humanSize(bytes) {
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const sizes = ['B', 'KB', 'MB', 'GB']
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + sizes[i]
}
function normLabel(s) {
  if (s == null) return ''
  return String(s)
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}
function ucfirst(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}
function fromMaybeStringNumber(v) {
  if (typeof v === 'number') return v
  if (typeof v === 'string') {
    const x = parseFloat(v)
    return Number.isFinite(x) ? x : null
  }
  return null
}

// Decide tumor/notumor from a normalized text label
function decideTumorByText(labelNorm) {
  if (!labelNorm) return null

  // explicit negatives
  const negExact = new Set([
    'no tumor',
    'no tumour',
    'no_tumor',
    'notumor',
    'normal',
    'negative',
    'healthy',
    'no mass',
    'no abnormality',
    'no brain tumor',
    'no brain tumour',
  ])
  if (negExact.has(labelNorm)) return false

  // patterns
  if (/^no\s+(brain\s+)?tumou?r$/.test(labelNorm)) return false
  if (
    /(^|\s)(no|without|absence\s+of)\s+(brain\s+)?tumou?r(\s|$)/.test(labelNorm)
  )
    return false

  // positives
  if (
    /(tumou?r|lesion|mass|glioma|meningioma|pituitary|gbm|lgg|astrocytoma|oligodendroglioma|metastasis)/.test(
      labelNorm
    )
  ) {
    return true
  }

  return null
}

/**
 * Interpret the model response and produce:
 *  - labelText (string)
 *  - labelNorm (normalized string)
 *  - isTumor (true/false/null)
 *  - confPct (0..100 or null)
 *
 * Special handling for:
 * - label_name, label_id
 * - probability_tumor (flip to 1-p for no_tumor cases)
 * - threshold (if provided, used when label is ambiguous)
 */
function interpretPrediction(data) {
  let labelText = null
  let labelNorm = null
  let isTumor = null
  let confPct = null
  let bestFromScores = null

  // --- Primary: label_name ---
  if (typeof data?.label_name === 'string' && data.label_name.length) {
    labelText = String(data.label_name)
    labelNorm = normLabel(labelText)
  }

  // --- Fallback label strings ---
  const labelCandidates = [
    data?.label,
    data?.class,
    data?.prediction,
    data?.predicted_label,
    data?.result,
    data?.verdict,
  ].filter((v) => typeof v === 'string' && v.length)
  if (!labelText && labelCandidates.length) {
    labelText = String(labelCandidates[0])
    labelNorm = normLabel(labelText)
  }

  // --- Numeric label_id convention: 1=tumor, 0=no_tumor (adjust if your model differs) ---
  if (
    isTumor === null &&
    (typeof data?.label_id === 'number' || typeof data?.label_id === 'string')
  ) {
    const id = parseInt(data.label_id, 10)
    if (id === 1) isTumor = true
    else if (id === 0) isTumor = false

    if (!labelText) {
      labelText = isTumor ? 'tumor' : 'no_tumor'
      labelNorm = normLabel(labelText)
    }
  }

  // --- Numeric prediction index + class names ---
  if (
    typeof data?.prediction === 'number' ||
    (typeof data?.prediction === 'string' && !isNaN(+data.prediction))
  ) {
    const idx = parseInt(data.prediction, 10)
    const nameArrays = [
      data?.classes,
      data?.labels,
      data?.class_names,
      data?.classNames,
    ].find((a) => Array.isArray(a) && a.length)
    if (nameArrays && nameArrays[idx]) {
      labelText = String(nameArrays[idx])
      labelNorm = normLabel(labelText)
    }
  }

  // --- Confidence from simple fields ---
  const confCandidates = [
    data?.confidence,
    data?.score,
    data?.probability,
    data?.prob,
    data?.p,
    data?.conf,
    data?.confidence_score,
  ]
  for (const c of confCandidates) {
    const v = fromMaybeStringNumber(c)
    if (v != null) {
      confPct =
        v <= 1
          ? Math.max(0, Math.min(1, v)) * 100
          : Math.max(0, Math.min(100, v))
      break
    }
  }

  // --- Bags of scores/probs ---
  const bag =
    data?.scores ||
    data?.probs ||
    data?.probabilities ||
    data?.softmax ||
    data?.class_scores ||
    data?.confidences
  if (!confPct && bag) {
    if (Array.isArray(bag)) {
      if (bag.length) {
        if (typeof bag[0] === 'number') {
          const max = Math.max(...bag)
          const idx = bag.indexOf(max)
          confPct = max <= 1 ? max * 100 : Math.min(100, max)
          const nameArrays = [
            data?.classes,
            data?.labels,
            data?.class_names,
            data?.classNames,
          ].find((a) => Array.isArray(a) && a.length)
          if (nameArrays && nameArrays[idx])
            bestFromScores = String(nameArrays[idx])
        } else if (typeof bag[0] === 'object' && bag[0] !== null) {
          let best = null
          for (const it of bag) {
            const s = fromMaybeStringNumber(
              it?.score ?? it?.confidence ?? it?.prob
            )
            if (s == null) continue
            if (!best || s > best.score)
              best = {
                label: it?.label ?? it?.class ?? it?.name ?? null,
                score: s,
              }
          }
          if (best) {
            confPct =
              best.score <= 1 ? best.score * 100 : Math.min(100, best.score)
            if (best.label) bestFromScores = String(best.label)
          }
        }
      }
    } else if (typeof bag === 'object') {
      let bestKey = null,
        bestVal = -Infinity
      for (const [k, vRaw] of Object.entries(bag)) {
        const v = fromMaybeStringNumber(vRaw)
        if (v == null) continue
        if (v > bestVal) {
          bestVal = v
          bestKey = k
        }
      }
      if (bestKey != null) {
        confPct = bestVal <= 1 ? bestVal * 100 : Math.min(100, bestVal)
        bestFromScores = String(bestKey)
      }
    }
  }

  if (bestFromScores) {
    labelText = bestFromScores
    labelNorm = normLabel(bestFromScores)
  }

  // --- Dedicated tumor probability + optional threshold ---
  let pTumor = fromMaybeStringNumber(
    data?.probability_tumor ??
      data?.tumor_probability ??
      data?.prob_tumor ??
      data?.p_tumor
  )
  // If only a "no tumor" prob is given, flip it.
  if (pTumor == null) {
    const pNoTumor = fromMaybeStringNumber(
      data?.probability_no_tumor ??
        data?.probability_normal ??
        data?.p_no_tumor ??
        data?.probability_negative
    )
    if (pNoTumor != null) {
      pTumor =
        pNoTumor <= 1 ? 1 - pNoTumor : (100 - Math.min(100, pNoTumor)) / 100
    }
  }

  const threshold = fromMaybeStringNumber(data?.threshold) // e.g., 0.05
  if (pTumor != null) {
    const pPct = pTumor <= 1 ? pTumor * 100 : Math.min(100, pTumor)

    const negAliases = new Set([
      'no tumor',
      'no tumour',
      'no_tumor',
      'notumor',
      'normal',
      'negative',
      'healthy',
    ])
    const labelSaysTumor = labelNorm
      ? !negAliases.has(labelNorm) &&
        /tumou?r|glioma|meningioma|pituitary|gbm|lgg/.test(labelNorm)
      : null

    if (labelSaysTumor === true || isTumor === true) {
      // Label indicates tumor → use p(tumor)
      confPct = pPct
    } else if (labelSaysTumor === false || isTumor === false) {
      // Label indicates no tumor → use 1 - p(tumor)
      confPct = Math.max(0, Math.min(100, 100 - pPct))
    } else if (typeof threshold === 'number' && Number.isFinite(threshold)) {
      // Use threshold if provided (compare against pTumor)
      const thrPct = threshold <= 1 ? threshold * 100 : threshold
      if (pPct >= thrPct) {
        isTumor = true
        confPct = pPct
        if (!labelText) {
          labelText = 'tumor'
          labelNorm = 'tumor'
        }
      } else {
        isTumor = false
        confPct = 100 - pPct
        if (!labelText) {
          labelText = 'no_tumor'
          labelNorm = 'no tumor'
        }
      }
    } else {
      // No clear label or threshold → decide by majority probability
      if (pPct >= 50) {
        isTumor = true
        confPct = pPct
        if (!labelText) {
          labelText = 'tumor'
          labelNorm = 'tumor'
        }
      } else {
        isTumor = false
        confPct = 100 - pPct
        if (!labelText) {
          labelText = 'no_tumor'
          labelNorm = 'no tumor'
        }
      }
    }
  }

  // If still unknown, try text-based decision
  if (isTumor === null && labelNorm) {
    const decided = decideTumorByText(labelNorm)
    if (decided !== null) isTumor = decided
  }

  return {
    labelText: labelText || 'Prediction',
    labelNorm: labelNorm || '',
    isTumor, // true/false/null
    confPct: confPct != null ? Math.round(confPct) : null, // 0..100
  }
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

/* ============================ File Handling =========================== */
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

/* ============================ Event Wiring ============================ */
// Open dialog
dropzone.addEventListener('click', () => fileInput.click())
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click()
})

// Drag/drop styling
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

// Picker
fileInput.addEventListener('change', (e) =>
  fileChosen(e.target.files[0], 'picker')
)

// Paste
window.addEventListener('paste', (e) => {
  const item = [...e.clipboardData.items].find((i) =>
    i.type.startsWith('image/')
  )
  if (item) fileChosen(item.getAsFile(), 'paste')
})

// Clear
clearBtn.addEventListener('click', resetUI)

/* ============================== Predict ============================== */
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

    // Optional: keep raw JSON for debugging
    if (resultJson) resultJson.textContent = JSON.stringify(data, null, 2)

    const { labelText, labelNorm, isTumor, confPct } = interpretPrediction(data)

    // Latency
    const clientMs = Math.round(performance.now() - t0)
    const serverMs =
      typeof data.inference_ms === 'number' ? data.inference_ms : null

    // Decide final verdict class
    const verdictIsTumor =
      isTumor === true ||
      (isTumor === null && decideTumorByText(labelNorm) === true)

    resultHero.classList.remove('good', 'bad')
    resultHero.classList.add(verdictIsTumor ? 'bad' : 'good')

    // Icons
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

    heroIcon.innerHTML = verdictIsTumor ? alertSvg : checkSvg
    heroTitle.textContent = verdictIsTumor
      ? 'Tumor detected'
      : 'No tumor detected'

    // Subtitle: subtype if present in label
    const subtype = [
      'glioma',
      'meningioma',
      'pituitary',
      'gbm',
      'lgg',
      'astrocytoma',
      'oligodendroglioma',
    ].find((k) => labelNorm.includes(k))
    heroSubtitle.textContent = subtype
      ? `Subtype: ${ucfirst(subtype)}`
      : labelText && labelText !== 'Prediction'
      ? `Model label: ${labelText}`
      : 'Model result'

    // Metrics
    metricConfidence.textContent = confPct != null ? `${confPct}%` : '—'
    metricLatency.textContent = serverMs
      ? `${serverMs} ms (server)`
      : `${clientMs} ms (round-trip)`

    resultHero.classList.remove('d-none')

    gaEvent('predict_success', {
      label: labelText,
      is_tumor: verdictIsTumor,
      confidence_pct: confPct ?? undefined,
      roundtrip_ms: clientMs,
    })

    // If still ambiguous, show debug toggle
    if (isTumor === null && confPct == null) {
      debugDetails?.classList.remove('d-none')
    }
  } catch (err) {
    showToast('Prediction failed: ' + err.message)
    resultHero.classList.add('d-none')
    gaEvent('predict_error', { message: String(err.message || err) })
  } finally {
    spinner.classList.add('d-none')
    predictBtn.disabled = false
  }
})
