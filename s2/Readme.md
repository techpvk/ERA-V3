# Animal Selector and File Uploader

This project is a simple web application that allows users to select an animal and upload a file. The application is built using Flask for the backend and HTML/CSS/JavaScript for the frontend.

## Features

- **Animal Selector**: Choose between three animals (Cat, Dog, Elephant) and see an image of the selected animal.
- **File Uploader**: Upload any file and receive its name, size, and type.
- **Responsive Design**: The layout is responsive and adapts to different screen sizes.
- **Animations**: The UI includes smooth animations for a better user experience.

## Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: CSS with animations and transitions

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/animal-selector-uploader.git
   cd animal-selector-uploader
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:5000/` to view the application.

## Project Structure
project/
│
├── app.py
├── static/
│ ├── images/
│ │ ├── cat.jpg
│ │ ├── dog.jpg
│ │ └── elephant.jpg
│ ├── style.css
│ └── script.js
├── templates/
│ └── index.html
└── requirements.txt

## Usage

- **Select an Animal**: Click on one of the radio buttons to select an animal. The corresponding image will be displayed.
- **Upload a File**: Click the "Upload" button after selecting a file to see its details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
