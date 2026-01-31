# Contributing to UPI Fraud Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

Feature suggestions are welcome! Please:
- Check existing issues first
- Provide clear use case
- Explain expected behavior

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/upi-fraud-detection.git
   cd upi-fraud-detection
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable

4. **Test your changes**
   ```bash
   python train.py  # Ensure training works
   python predict.py  # Ensure inference works
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Testing

Before submitting PR:
- Ensure all existing functionality works
- Add tests for new features
- Run `python train.py` successfully
- Test `predict.py` with sample data

## Questions?

Feel free to open an issue for any questions!

---

Thank you for contributing! 🎉
