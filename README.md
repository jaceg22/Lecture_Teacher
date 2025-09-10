# Lecture Teacher 📚

An AI-powered educational tool that transforms various document formats into interactive learning experiences with summaries, visual teaching aids, and quizzes.

## Features ✨

- **Multi-Format Support**: Process documents in various formats:
  - PowerPoint presentations (`.pptx`)
  - PDF documents (`.pdf`)
  - Images (`.png`, `.jpg`, `.jpeg`)
  - Word documents (`.docx`)

- **Intelligent Content Processing**:
  - Extract and process text from documents using OCR
  - Generate comprehensive summaries
  - Create educational content with visual aids

- **Interactive Learning**:
  - Generate multiple choice questions with feedback
  - Create true/false quizzes with explanations
  - Use images from presentations/documents for enhanced teaching

- **AI-Powered**: Leverages Groq API for natural language processing and content generation

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/jaceg22/Lecture_Teacher.git
cd Lecture_Teacher
```

2. Install required dependencies:
```bash
pip install streamlit groq pytesseract pillow python-dotenv PyPDF2 python-pptx python-docx torch diffusers PyMuPDF
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Install Tesseract OCR:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage 🚀

1. Start the Streamlit application:
```bash
streamlit run imagesummary.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload your document (PPTX, PDF, PNG, JPG, or DOCX)

4. The application will:
   - Extract text content using OCR
   - Generate educational summaries
   - Create interactive quizzes
   - Provide visual teaching aids

## How It Works 🔧

1. **Document Processing**: The app accepts various file formats and extracts text content using appropriate libraries and OCR
2. **Content Analysis**: Uses AI to analyze and understand the extracted content
3. **Educational Content Generation**: Creates summaries, explanations, and teaching materials
4. **Quiz Generation**: Automatically generates multiple choice and true/false questions with detailed feedback
5. **Visual Integration**: Incorporates images from the original documents to enhance learning

## Requirements 📋

- Python 3.7+
- Groq API key
- Tesseract OCR
- Streamlit
- Required Python packages (see installation section)

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is open source and available under the [MIT License](LICENSE).

## Support 💬

If you encounter any issues or have questions, please open an issue on GitHub.