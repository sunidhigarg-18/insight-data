document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statePills = document.querySelectorAll(".state-pill");
  const viewPanels = {
    "initial": document.getElementById("view-initial"),
    "dataset-preview": document.getElementById("view-dataset-preview"),
    "split-results": document.getElementById("view-split-results"),
  };

  const tabDataset = document.getElementById("tabDataset");
  const tabSolution = document.getElementById("tabSolution");
  const emptyState = document.getElementById("emptyState");
  const exampleChips = document.getElementById("exampleChips");
  const clearBtn = document.getElementById("clearBtn");

  const plusBtn = document.getElementById("plusBtn");
  const uploadMenu = document.getElementById("uploadMenu");
  const uploadOption = document.getElementById("uploadOption");
  const datasetFileInput = document.getElementById("datasetFileInput");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const closeBtns = document.querySelectorAll(".panel-close-btn");

  // Toast notification elements
  const toastNotification = document.getElementById("toastNotification");
  const toastCloseBtn = document.getElementById("toastCloseBtn");

  let electricityChartInstance = null;
  let toastTimer = null;

  // UI state: drives tab availability and the empty-state message
  let hasDataset = false;
  let hasResults = false;

  function updateTabs() {
    if (tabDataset) tabDataset.disabled = !hasDataset;
    if (tabSolution) tabSolution.disabled = !hasResults;
  }

  function updateEmptyState() {
    if (emptyState) emptyState.classList.toggle("hidden", !(hasDataset && !hasResults));
  }

  // Clear (×) button visibility follows the input content
  function updateClearBtn() {
    if (clearBtn) clearBtn.classList.toggle("hidden", questionInput.value.length === 0);
  }

  // Escape HTML helper for table rendering security
  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Toast Notification System
  function showToast(type, title, message, duration = 5000) {
    if (!toastNotification) return;

    const toastIcon = document.getElementById("toastIcon");
    const toastTitle = document.getElementById("toastTitle");
    const toastMessage = document.getElementById("toastMessage");

    toastNotification.className = `toast-notification ${type}`;
    if (toastTitle) toastTitle.textContent = title;
    if (toastMessage) toastMessage.textContent = message;

    if (toastIcon) {
      if (type === "success") {
        toastIcon.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>`;
      } else {
        toastIcon.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>`;
      }
    }

    toastNotification.classList.remove("hidden");

    if (toastTimer) clearTimeout(toastTimer);
    if (duration > 0) {
      toastTimer = setTimeout(() => {
        toastNotification.classList.add("hidden");
      }, duration);
    }
  }

  if (toastCloseBtn) {
    toastCloseBtn.addEventListener("click", () => {
      if (toastNotification) toastNotification.classList.add("hidden");
    });
  }

  // Function to switch view panels
  function switchView(targetState) {
    // Update active pill
    statePills.forEach(pill => {
      pill.classList.toggle("active", pill.dataset.state === targetState);
    });

    // Update active view
    Object.keys(viewPanels).forEach(key => {
      const panel = viewPanels[key];
      if (panel) {
        panel.classList.toggle("active-view", key === targetState);
      }
    });

    // Resize chart when the Solution view becomes visible
    if (targetState === "split-results" && electricityChartInstance) {
      setTimeout(() => electricityChartInstance && electricityChartInstance.resize(), 50);
    }
  }

  // Bind state switcher buttons
  statePills.forEach(pill => {
    pill.addEventListener("click", () => {
      const targetState = pill.dataset.state;
      switchView(targetState);
    });
  });

  // Render Dataset Preview from actual Pandas DataFrame response
  function renderDatasetPreview(dataset) {
    const fileNameEl = document.getElementById("datasetFileName");
    const fileTypeEl = document.getElementById("datasetFileType");
    const fileSpecsEl = document.getElementById("datasetFileSpecs");
    const tableHead = document.getElementById("previewTableHead");
    const tableBody = document.getElementById("previewTableBody");

    if (fileNameEl) fileNameEl.textContent = dataset.filename;
    if (fileTypeEl) fileTypeEl.textContent = dataset.file_type;
    if (fileSpecsEl) {
      fileSpecsEl.textContent = `${dataset.file_size} • ${dataset.cols} columns • ${dataset.rows} rows`;
    }

    // Populate actual columns and rows
    if (tableHead && dataset.columns) {
      tableHead.innerHTML = `<tr>${dataset.columns.map(col => `<th>${escapeHtml(col)}</th>`).join("")}</tr>`;
    }

    if (tableBody && dataset.preview) {
      tableBody.innerHTML = dataset.preview.map(row => {
        return `<tr>${dataset.columns.map(col => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`;
      }).join("");
    }
  }

  // Build example-question chips from the dataset's columns (falls back to generic ones)
  function buildExampleQuestions(dataset) {
    const generic = [
      "Which category has the highest total?",
      "Compare values across different periods.",
      "What trends can you identify in this dataset?"
    ];
    try {
      const cols = dataset.columns || [];
      const rows = dataset.preview || [];
      if (cols.length < 2 || rows.length === 0) return generic;

      const clean = name => String(name).replace(/\s*\(.*?\)\s*/g, " ").replace(/\s+/g, " ").trim();
      const isNumeric = col => rows.every(r => {
        const v = r[col];
        return v !== null && v !== undefined && String(v).trim() !== "" && Number.isFinite(Number(v));
      });
      const isDateCol = col =>
        /date|time|month|year|day|period/i.test(col) ||
        rows.every(r => /^\d{4}-\d{2}(-\d{2})?/.test(String(r[col])));

      const dateCol = cols.find(c => isDateCol(c));
      const numericCols = cols.filter(c => c !== dateCol && isNumeric(c));
      const categoryCol = cols.find(c => c !== dateCol && !numericCols.includes(c));

      if (numericCols.length === 0 || !categoryCol) return generic;

      const num = clean(numericCols[0]).toLowerCase();
      const cat = clean(categoryCol);
      const num2 = numericCols[1] ? clean(numericCols[1]).toLowerCase() : num;

      return [
        `Which ${cat} has the highest ${num}?`,
        dateCol ? `Show the trend in ${num2} over time.` : `Compare ${num2} across different ${cat} values.`,
        `What is the total ${num} for all ${cat} values?`
      ];
    } catch (e) {
      return generic;
    }
  }

  function renderExampleChips(dataset) {
    if (!exampleChips) return;
    exampleChips.innerHTML = buildExampleQuestions(dataset)
      .map(q => `<button type="button" class="example-chip" data-prompt="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
      .join("");
  }

  // One delegated handler: chips only fill the input (never auto-submit)
  if (exampleChips) {
    exampleChips.addEventListener("click", (e) => {
      const chip = e.target.closest(".example-chip");
      if (!chip) return;
      questionInput.value = chip.dataset.prompt || "";
      updateClearBtn();
      questionInput.focus();
    });
  }

  // Requirement: Make the existing "+" button open a file picker
  plusBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    // Directly trigger file picker
    datasetFileInput.click();
  });

  // Also allow clicking the "Upload Dataset" option in the menu
  if (uploadOption) {
    uploadOption.addEventListener("click", (e) => {
      e.stopPropagation();
      uploadMenu.classList.add("hidden");
      datasetFileInput.click();
    });
  }

  // Handle file selection and upload
  datasetFileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate extension on client side as an immediate pre-check
    const validExtensions = [".csv", ".xlsx", ".xls"];
    const fileNameLower = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileNameLower.endsWith(ext));

    if (!isValid) {
      showToast("error", "Unsupported File", "Unsupported file type. Please upload a CSV or Excel file.");
      datasetFileInput.value = "";
      return;
    }

    // Upload via FormData to Flask backend
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (response.ok && data.success && data.dataset) {
        // Success: update preview with actual pandas data
        renderDatasetPreview(data.dataset);
        renderExampleChips(data.dataset);
        hasDataset = true;
        hasResults = false; // a new dataset invalidates any previous result
        updateTabs();
        updateEmptyState();
        switchView("dataset-preview");
        showToast("success", "Dataset Active", `${data.dataset.filename} uploaded successfully (${data.dataset.rows} rows).`);
      } else {
        // Error: keep previous valid dataset active
        showToast("error", "Upload Failed", data.error || "Unable to read dataset. Previous active dataset is preserved.");
      }
    } catch (err) {
      showToast("error", "Upload Error", "Server could not process file. Previous active dataset is preserved.");
    } finally {
      // Clear input so selecting the same file again triggers change event
      datasetFileInput.value = "";
    }
  });

  // Check if session already has an active dataset on load
  async function checkActiveDataset() {
    try {
      const res = await fetch("/api/active_dataset");
      if (res.ok) {
        const data = await res.json();
        if (data.has_dataset && data.dataset) {
          renderDatasetPreview(data.dataset);
          renderExampleChips(data.dataset);
          hasDataset = true;
          updateTabs();
          updateEmptyState();
          // Land on the dataset view only if the user hasn't moved on already
          if (viewPanels["initial"].classList.contains("active-view")) {
            switchView("dataset-preview");
          }
        }
      }
    } catch (e) {
      // Silent catch on init
    }
  }
  checkActiveDataset();

  // Close buttons on panels: hop to the other available view, else the start screen
  closeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const panel = btn.closest(".view-panel");
      const fromSolution = panel && panel.id === "view-split-results";
      if (fromSolution && hasDataset) switchView("dataset-preview");
      else if (!fromSolution && hasResults) switchView("split-results");
      else switchView("initial");
    });
  });

  // Clear (×) button + input listeners
  questionInput.addEventListener("input", updateClearBtn);
  if (clearBtn) {
    // Keep focus in the input when pressing the button
    clearBtn.addEventListener("mousedown", (e) => e.preventDefault());
    clearBtn.addEventListener("click", () => {
      questionInput.value = "";
      updateClearBtn();
      questionInput.focus();
    });
  }
  updateClearBtn();
  updateTabs();

  // Render Dynamic Calculation Details and Supporting Data
  function renderCalculationDetails(calcDetails, dataUsed, highlightIdx = -1) {
    const container = document.getElementById("calculationContent");
    if (!container) return;

    let html = "";
    if (calcDetails) {
      html += `
        <div class="calc-step-block">
          <div class="step-title">${escapeHtml(calcDetails.method || "Analytical Method")}</div>
          ${calcDetails.formula ? `
            <div class="formula-line">
              <span class="formula-val highlighted">${escapeHtml(calcDetails.formula)}</span>
            </div>
          ` : ""}
          ${calcDetails.steps && calcDetails.steps.length > 0 ? `
            <div class="calc-steps-list" style="margin-top: 8px; display: flex; flex-direction: column; gap: 6px;">
              ${calcDetails.steps.map(step => `
                <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">${escapeHtml(step)}</div>
              `).join("")}
            </div>
          ` : ""}
        </div>
      `;
    }

    if (dataUsed && dataUsed.length > 0) {
      const keys = Object.keys(dataUsed[0]);
      html += `
        <div class="calc-step-block" style="margin-top: 14px;">
          <div class="step-title">Supporting Data (${dataUsed.length} records)</div>
          <div class="table-container" style="max-height: 220px; overflow-y: auto;">
            <table class="calc-table">
              <thead>
                <tr>${keys.map(k => `<th>${escapeHtml(k)}</th>`).join("")}</tr>
              </thead>
              <tbody>
                ${dataUsed.slice(0, 15).map((row, idx) => `
                  <tr class="${idx === highlightIdx ? 'highlight-row' : ''}">
                    ${keys.map(k => `<td>${idx === highlightIdx && k === keys[0] ? '<span class="active-dot"></span>' : ''}${escapeHtml(String(row[k]))}</td>`).join("")}
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    container.innerHTML = html;
  }

  // Render Dynamic Analysis Chart (Bar, Line, or hidden)
  function renderAnalysisChart(chartData) {
    const ctx = document.getElementById("electricityChart");
    const chartWrapper = document.getElementById("chartWrapper");
    const chartTitle = document.getElementById("chartTitle");

    if (!ctx) return;

    if (!chartData || !chartData.labels || chartData.labels.length === 0) {
      if (chartWrapper) chartWrapper.style.display = "none";
      return;
    }

    if (chartWrapper) chartWrapper.style.display = "block";
    if (chartTitle && chartData.title) chartTitle.textContent = chartData.title;

    if (electricityChartInstance) {
      electricityChartInstance.destroy();
    }

    const labels = chartData.labels;
    const values = chartData.values;
    const chartType = chartData.type === "line" ? "line" : "bar";
    const highlightIdx = chartData.highlight_index ?? -1;

    let datasetConfig = {};
    if (chartType === "line") {
      datasetConfig = {
        label: chartData.y_label || "Value",
        data: values,
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56, 189, 248, 0.15)",
        fill: true,
        tension: 0.3,
        pointBackgroundColor: "#38bdf8",
        pointBorderColor: "#ffffff",
        pointHoverRadius: 6,
        pointRadius: 4
      };
    } else {
      const backgroundColors = labels.map((_, i) =>
        i === highlightIdx ? "rgba(56, 189, 248, 1.0)" : "rgba(2, 132, 199, 0.7)"
      );
      const borderColors = labels.map((_, i) =>
        i === highlightIdx ? "#38bdf8" : "rgba(56, 189, 248, 0.4)"
      );
      datasetConfig = {
        label: chartData.y_label || "Value",
        data: values,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 6,
        barPercentage: 0.6
      };
    }

    electricityChartInstance = new Chart(ctx, {
      type: chartType,
      data: {
        labels: labels,
        datasets: [datasetConfig]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f172a",
            titleColor: "#f8fafc",
            bodyColor: "#38bdf8",
            borderColor: "rgba(56, 189, 248, 0.3)",
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            grid: { display: false, drawBorder: false },
            ticks: {
              color: "#94a3b8",
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 }
            }
          },
          y: {
            beginAtZero: true,
            grid: { color: "rgba(255, 255, 255, 0.05)", drawBorder: false },
            ticks: {
              color: "#64748b",
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 }
            }
          }
        }
      }
    });
  }

  // Send button action: queries /ask backend endpoint
  sendBtn.addEventListener("click", async () => {
    const query = questionInput.value.trim();
    if (!query) {
      questionInput.focus();
      return;
    }

    sendBtn.disabled = true;
    sendBtn.style.opacity = "0.5";

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query })
      });

      const data = await res.json();

      if (res.ok && data) {
        const directAnswerEl = document.getElementById("directAnswerText");
        const insightCalloutText = document.getElementById("insightCalloutText");

        if (directAnswerEl) directAnswerEl.textContent = data.answer;
        if (insightCalloutText) insightCalloutText.textContent = data.answer;

        renderCalculationDetails(
          data.calculation_details,
          data.data_used,
          data.chart_data?.highlight_index ?? -1
        );

        hasResults = true;
        updateTabs();
        updateEmptyState();
        switchView("split-results");
        renderAnalysisChart(data.chart_data);

        if (!data.success) {
          showToast("error", "Question Clarification", data.answer);
        }
      } else {
        showToast("error", "Analysis Error", data.error || data.answer || "Could not analyze question.");
      }
    } catch (err) {
      showToast("error", "Network Error", "Could not connect to the question-analysis service.");
    } finally {
      sendBtn.disabled = false;
      sendBtn.style.opacity = "1";
    }
  });

  // Enter key in question input
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendBtn.click();
    }
  });

});
