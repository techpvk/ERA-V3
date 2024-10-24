# Daily Learning Dashboard Chrome Extension

## Overview

The Daily Learning Dashboard is a Chrome extension designed to help users save and organize their daily learning activities. It allows users to save selected text from web pages, bookmark entire web pages, and view their saved content in a structured dashboard.

## Features

1. **Save Selected Text**: Users can select any text on a webpage, right-click, and save it to their dashboard along with the source URL, a description, and tags.

2. **Save Webpages**: Users can save entire webpages with the URL, a custom description, and tags.

3. **Tagging System**: Both selected text and saved webpages can be tagged for easy categorization and searching.

4. **Dashboard View**: A comprehensive dashboard displays all saved items, organized by date and searchable by tags. Each item shows its source URL, content (for selected text), description, and tags.

5. **Uniform URL Display**: All saved items, whether selected text or webpages, display their source URL in a consistent manner for easy reference and access.

## Installation

1. Clone this repository or download the source code.
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Enable "Developer mode" in the top right corner.
4. Click "Load unpacked" and select the directory containing the extension files.

## Usage

### Saving Selected Text

1. Select text on any webpage.
2. Right-click and choose "Save to Learning Dashboard" from the context menu.
3. In the popup, add a description and tags (comma-separated).
4. Click "Save Selection".

### Saving Webpages

1. Click the extension icon in the Chrome toolbar.
2. The current webpage URL will be auto-filled.
3. Add a description and tags (comma-separated).
4. Click "Save Webpage".

### Viewing the Dashboard

1. Click the extension icon in the Chrome toolbar.
2. Click "Open Dashboard".
3. View your saved items, organized by date.
4. Use the search bar to filter items by tags.

## File Structure

- `manifest.json`: Extension configuration file
- `background.js`: Handles background processes and context menu
- `content.js`: Manages content script for text selection
- `popup.html` & `popup.js`: Handles the extension popup for saving webpages
- `selection_popup.html` & `selection_popup.js`: Manages the popup for saving selected text
- `dashboard.html` & `dashboard.js`: Implements the dashboard view and functionality

## Technologies Used

- HTML5
- CSS3
- JavaScript (ES6+)
- Chrome Extension API

## Known Issues

- The GitHub integration for preserving code formatting is currently not functioning as intended. We are working on improving this feature in future updates.

## Future Enhancements

- Improve GitHub code block handling and formatting preservation
- Implement data sync across devices
- Add export/import functionality for saved data
- Integrate with popular note-taking applications

## Contributors

Venkatakumar Puvvada

## License

