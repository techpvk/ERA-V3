document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  
  document.getElementById('searchButton').addEventListener('click', () => {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    loadDashboard(searchTerm);
  });

  document.getElementById('exportButton').addEventListener('click', exportData);
  document.getElementById('pdfExportButton').addEventListener('click', exportPDF);
});

function loadDashboard(searchTerm = '') {
  chrome.storage.local.get({ learnings: [] }, (result) => {
    const dashboard = document.getElementById('dashboard');
    dashboard.innerHTML = '';

    // Sort learnings by timestamp in descending order (latest first)
    const sortedLearnings = result.learnings.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    const groupedLearnings = groupByDate(sortedLearnings);

    Object.keys(groupedLearnings).forEach(date => {
      const dateSection = document.createElement('div');
      dateSection.className = 'date-section';
      
      const dateHeader = document.createElement('div');
      dateHeader.className = 'date-header';
      dateHeader.textContent = formatDate(date);
      dateSection.appendChild(dateHeader);

      let hasVisibleItems = false;

      groupedLearnings[date].forEach(item => {
        if (!searchTerm || (item.tags && item.tags.some(tag => tag.toLowerCase().includes(searchTerm)))) {
          const itemElement = createItemElement(item);
          dateSection.appendChild(itemElement);
          hasVisibleItems = true;
        }
      });

      if (hasVisibleItems) {
        dashboard.appendChild(dateSection);
      }
    });

    if (dashboard.children.length === 0) {
      dashboard.innerHTML = '<p>No items found.</p>';
    }
  });
}

function groupByDate(learnings) {
  return learnings.reduce((groups, item) => {
    const date = new Date(item.timestamp).toLocaleDateString();
    if (!groups[date]) {
      groups[date] = [];
    }
    groups[date].push(item);
    return groups;
  }, {});
}

function formatDate(dateString) {
  const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  return new Date(dateString).toLocaleDateString('en-US', options);
}

function createItemElement(item) {
  const itemElement = document.createElement('div');
  itemElement.className = 'learning-item';

  let contentHtml = '';
  if (item.type === 'selection') {
    contentHtml = `
      <h3>Selected Text</h3>
      <a href="${item.url}" target="_blank">${item.url}</a>
      <div><strong>Content:</strong> <div class="selected-content">${item.html || item.content}</div></div>
      <p><strong>Description:</strong> ${item.description}</p>
    `;
  } else if (item.type === 'webpage') {
    contentHtml = `
      <h3>Saved Webpage</h3>
      <a href="${item.url}" target="_blank">${item.url}</a>
      <p><strong>Description:</strong> ${item.description}</p>
    `;
  }

  const tagsHtml = item.tags && item.tags.length > 0 
    ? `<div><strong>Tags:</strong> ${item.tags.map(tag => `<span class="tag">${tag}</span>`).join(' ')}</div>` 
    : '';

  itemElement.innerHTML = `
    ${contentHtml}
    ${tagsHtml}
    <p><small>${new Date(item.timestamp).toLocaleString()}</small></p>
  `;

  return itemElement;
}

function exportData() {
  chrome.storage.local.get({ learnings: [] }, (result) => {
    const dataStr = JSON.stringify(result.learnings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'learning_dashboard_export.json';

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  });
}

function exportPDF() {
  // Ensure jsPDF is loaded
  if (typeof window.jspdf === 'undefined') {
    console.error('jsPDF library not loaded');
    alert('PDF export is not available. Please try again later.');
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  
  chrome.storage.local.get({ learnings: [] }, (result) => {
    let yOffset = 10;

    doc.setFontSize(20);
    doc.text('Learning Dashboard Report', 105, yOffset, { align: 'center' });
    yOffset += 20;

    doc.setFontSize(12);
    result.learnings.forEach((item, index) => {
      if (yOffset > 280) {
        doc.addPage();
        yOffset = 10;
      }

      doc.setFontSize(14);
      doc.text(`${index + 1}. ${item.type === 'selection' ? 'Selected Text' : 'Saved Webpage'}`, 10, yOffset);
      yOffset += 10;

      doc.setFontSize(12);
      const urlLines = doc.splitTextToSize(`URL: ${item.url}`, 180);
      doc.text(urlLines, 15, yOffset);
      yOffset += urlLines.length * 7;

      if (item.type === 'selection') {
        const contentLines = doc.splitTextToSize(`Content: ${item.content || item.html}`, 180);
        doc.text(contentLines, 15, yOffset);
        yOffset += contentLines.length * 7;
      }

      const descriptionLines = doc.splitTextToSize(`Description: ${item.description}`, 180);
      doc.text(descriptionLines, 15, yOffset);
      yOffset += descriptionLines.length * 7;

      if (item.tags && item.tags.length > 0) {
        const tagLines = doc.splitTextToSize(`Tags: ${item.tags.join(', ')}`, 180);
        doc.text(tagLines, 15, yOffset);
        yOffset += tagLines.length * 7;
      }

      doc.text(`Saved on: ${new Date(item.timestamp).toLocaleString()}`, 15, yOffset);
      yOffset += 15;
    });

    try {
      doc.save('learning_dashboard_report.pdf');
    } catch (error) {
      console.error('Error saving PDF:', error);
      alert('An error occurred while generating the PDF. Please try again.');
    }
  });
}
