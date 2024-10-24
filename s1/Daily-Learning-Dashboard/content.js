chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getSelection") {
    let selection = window.getSelection();
    let html = "";
    let text = "";
    let url = window.location.href;

    if (selection.rangeCount > 0) {
      const range = selection.getRangeAt(0);
      
      // Check if we're on LinkedIn
      if (window.location.hostname.includes("linkedin.com")) {
        const postElement = range.commonAncestorContainer.closest('.feed-shared-update-v2, .occludable-update');
        if (postElement) {
          // Extract the post content
          const contentElement = postElement.querySelector('.feed-shared-update-v2__description, .feed-shared-text');
          if (contentElement) {
            html = contentElement.innerHTML;
            text = contentElement.textContent;
          }
          
          // Find the "Copy link to post" button
          const copyLinkButton = postElement.querySelector('button[aria-label="Copy link to post"]');
          if (copyLinkButton) {
            // Create a custom event to capture the copied URL
            const urlCaptured = new Promise(resolve => {
              document.addEventListener('copy', function onCopy(e) {
                document.removeEventListener('copy', onCopy);
                resolve(e.clipboardData.getData('text/plain'));
              });
            });

            // Click the "Copy link to post" button
            copyLinkButton.click();

            // Wait for the URL to be captured
            urlCaptured.then(capturedUrl => {
              url = capturedUrl;
              sendResponse({ html, text, url });
            });

            return true; // Indicates that the response is asynchronous
          }
        }
      } else if (window.location.hostname.includes("github.com")) {
        // Existing GitHub handling code
        const codeBlock = range.commonAncestorContainer.closest('.blob-wrapper, .highlight');
        if (codeBlock) {
          const codeLines = codeBlock.querySelectorAll('td.blob-code, .blob-code-inner');
          html = '<pre><code class="github-code-block">';
          codeLines.forEach(line => {
            html += line.outerHTML;
          });
          html += '</code></pre>';
          text = Array.from(codeLines).map(line => line.textContent).join('\n');
        }
      }

      // If not handled by LinkedIn or GitHub specific code
      if (!html && !text) {
        const div = document.createElement('div');
        div.appendChild(range.cloneContents());
        html = div.innerHTML;
        text = selection.toString();
      }
    }

    // If we haven't sent a response yet (for non-LinkedIn or LinkedIn without "Copy link to post" button)
    sendResponse({ html, text, url });
  }
});
