# Persona Classifier with Streamlit and Hugging Face 🚀

This project provides an interactive web app built with **Streamlit** and a fine-tuned **RoBERTa** model to classify personas based on a user's bio and social media posts. The app leverages **Transformers** and **PyTorch** for text classification and allows real-time feedback and model retraining.
![Demo](assets/demo_image.png)
![Demo](demo/demo.gif)

---

## 📋 Features

- **Persona Prediction**: Classifies user input into predefined personas (e.g., **Tech Enthusiast**, **Foodie Explorer**, **Fitness Buff**, etc.).
- **Real-Time Feedback**: Allows users to submit feedback and correct predictions. 🔄
- **Model Retraining**: Users can retrain the model with new feedback data, ensuring continuous learning. 🛠️
- **Customizable Background**: Set a background video/image with **Streamlit**. 🌄
- **Interactive UI**: Clean, user-friendly interface built with **Streamlit**. 🎨
- **Model Caching**: Efficient model loading for faster predictions. ⚡

---

## 🛠️ Technologies Used

- **Streamlit**: For building interactive web applications.
- **Transformers**: For text classification with **RoBERTa**.
- **PyTorch**: For deep learning-based persona classification.
- **Hugging Face Hub**: For model sharing (if hosted remotely).
- **pandas & NumPy**: For data manipulation and processing.
- **Scikit-learn**: For model evaluation and preprocessing.
- **Docker**: For containerizing the application for easy deployment.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.7+** installed.

---

# Installation

1.  **Clone the repository**:

    ```bash
    git clone [https://github.com//persona-classifier.git](https://github.com//persona-classifier.git)
    cd persona-classifier
    ```

2.  **Create a virtual environment**:

    ```bash
    python -m venv persona-env
    source persona-env/bin/activate  # On Windows: persona-env\Scripts\activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app**:

    ```bash
    streamlit run main.py
    ```

    The app will be accessible at `http://localhost:8501` in your browser. 🌐

---

## 🧪 Testing

This project includes comprehensive testing to ensure code quality and reliability.

### Quick Test Commands

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit      # Unit tests only
python run_tests.py --type integration  # Integration tests only
python run_tests.py --type e2e       # End-to-end tests only
python run_tests.py --type fast      # Fast tests (exclude slow ones)

# Run with additional checks
python run_tests.py --all-checks     # Tests + linting + type checking
```

### Test Coverage

The test suite provides comprehensive coverage:
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows
- **Coverage Reports**: HTML and XML coverage reports

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── e2e/                     # End-to-end tests
└── README.md               # Detailed testing documentation
```

For detailed testing information, see [tests/README.md](tests/README.md).

---

## Docker Deployment (Optional) 🐳

To run the app in a Docker container:

1.  **Pull the image from Docker Hub**:

    ```bash
    docker pull shuvamastro/persona-app
    ```

    This command will download the `persona-app` image from your Docker Hub repository.

2.  **Run the container**:

    ```bash
    docker run -p 8501:8501 shuvamastro/persona-app
    ```

    This command maps port `8501` from your Docker container to port `8501` on your host machine, which is the default port for Streamlit.

3.  **Visit the app**:

    Open a browser and go to `http://localhost:8501` to access the app.

---

## Retraining the Model 🔄

The app allows users to retrain the model with new feedback. After retraining, the app automatically loads the new model without requiring a restart. 🚀

To retrain the model, click the "**Retrain Model**" button after submitting feedback. This will train the model with your new data. 🧠

---

## Contributing 🤝

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or new features. If you encounter any issues, please report them in the "**Issues**" section. 📝

### Development Guidelines

1. **Write Tests**: All new features should include appropriate tests
2. **Run Tests**: Ensure all tests pass before submitting PRs
3. **Code Quality**: Follow the project's coding standards
4. **Documentation**: Update documentation for new features

---

## 📊 Project Status

![Tests](https://github.com/yourusername/persona-classifier/workflows/Tests/badge.svg)
![Coverage](https://codecov.io/gh/yourusername/persona-classifier/branch/main/graph/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

