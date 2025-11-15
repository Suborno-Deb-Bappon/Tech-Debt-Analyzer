function extractTextAndCode() {
  const question = document.querySelector('.question .post-layout .postcell .s-prose');
  if (!question) {
    console.error('Question text not found!');
    return { text: '', code: '' };
  }
  const codeBlocks = question.querySelectorAll('pre code');
  const text = question.innerText || '';
  const code = Array.from(codeBlocks).map(block => block.innerText || '').join('\n');
  return { text: text.trim(), code: code.trim() };
}

function addPredictButton() {
  const container = document.querySelector('.question .post-layout .postcell .s-prose');
  if (!container) {
    console.error('Could not find the question div to attach the button.');
    return;
  }

  const button = document.createElement('button');
  button.innerText = '⚡ Predict Technical Debt';
  button.style.cssText = `
    margin-top: 15px;
    padding: 12px 18px;
    background: linear-gradient(135deg, #f48024, #f29e4c);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 15px;
    cursor: pointer;
    transition: background 0.3s ease;
  `;
  button.onmouseover = () => button.style.background = 'linear-gradient(135deg,#f29e4c,#f48024)';
  button.onmouseout = () => button.style.background = 'linear-gradient(135deg,#f48024,#f29e4c)';

  const resultDiv = document.createElement('div');
  resultDiv.id = 'td-result-panel';
  resultDiv.style.cssText = `
    margin-top: 20px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: #f9f9f9;
    display: none;
    font-family: sans-serif;
  `;

  button.addEventListener('click', () => {
    const { text, code } = extractTextAndCode();
    if (!text && !code) {
      alert('No content found to analyze.');
      return;
    }
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `<em>⏳ Analyzing...</em>`;
    predictTechnicalDebt(text, code, resultDiv);
  });

  container.appendChild(button);
  container.appendChild(resultDiv);
}

function predictTechnicalDebt(text, code, resultDiv) {
  fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, code })
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        resultDiv.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
        return;
      }

      // Color map for TD types
      const colorMap = {
        Architecture:'#8e44ad', Build:'#27ae60', Code:'#3498db', Design:'#f39c12',
        Documentation:'#e67e22', Infrastructure:'#16a085', Test:'#c0392b',
        Requirements:'#d35400', Versioning:'#2c3e50'
      };
      const typeColor = colorMap[data.td_type] || '#555';

      const confPct = (data.confidence * 100).toFixed(1);

      resultDiv.innerHTML = `
        <div style="margin-bottom:15px;">
          <div style="font-size:17px; font-weight:bold; color:#333;">
            Prediction: ${data.prediction}
            <span style="font-size:14px; color:#666;">(${confPct}%)</span>
          </div>
          <div style="background:#eee; border-radius:4px; overflow:hidden; height:8px; margin-top:5px;">
            <div style="width:${confPct}%; height:100%; background:#4caf50;"></div>
          </div>
        </div>

        ${data.prediction.toLowerCase().includes('debt') ? `
          <div style="margin-bottom:15px;">
            <div style="margin-bottom:5px;">
              <span style="
                background:${typeColor};
                color:#fff;
                padding:3px 8px;
                border-radius:4px;
                font-size:14px;
              ">${data.td_type || '—'}</span>
            </div>
            <div style="font-size:14px; color:#333;">
              <strong>Why:</strong> ${data.td_type_rationale || '—'}
            </div>
          </div>

          <div>
            <strong>Top Evidence:</strong>
            <div style="margin-top:8px; display:flex; flex-direction:column; gap:6px;">
              ${data.rag_used.slice(0, 5).map((r, i) => `
                <details style="background:#fff; border:1px solid #ddd; border-radius:4px; padding:6px;">
                  <summary style="cursor:pointer; font-weight:500;">
                    [${i+1}] ${r.label} (sim=${r.similarity?.toFixed(3) ?? 'n/a'})
                  </summary>
                  <div style="margin-top:4px; color:#444; font-size:13px;">
                    ${r.body_snippet}
                  </div>
                </details>
              `).join('')}
            </div>
          </div>
        ` : ''}
      `;
    })
    .catch(err => {
      console.error('Error:', err);
      resultDiv.innerHTML = `<span style="color:red;">Failed to get prediction from the local API.</span>`;
    });
}

addPredictButton();
