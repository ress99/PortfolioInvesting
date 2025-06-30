# PortfolioInvesting

**PortfolioInvesting** provides a collection of classes and functions designed to support the development of investment strategies using Computational Finance methods.

---

## Code Installation

```bash
git clone https://github.com/ress99/PortfolioInvesting.git
cd PortfolioInvesting
```

---

## Dependency Installation

```bash
poetry install --no-root
poetry run pip install PyQt5==5.15.9
```

`PyQt5` is installed separately due to compatibility issues with Poetry.

---

## Running the Project

### In **VS Code**

1. Open the folder in VS Code.
2. Press `Ctrl+Shift+P` and select **Python: Select Interpreter**.
3. Choose the interpreter created by Poetry. You can find its path with:

```bash
poetry env info --path
```

---

### In the **Terminal**

1. Get the virtual environment path:

```bash
poetry env info --path
```

2. Activate it:

```bash
<path>\Scripts\activate
```

3. Run one of the available scripts:

- To run the GUI interface:

```bash
python gui.py
```

- To run a CLI demo:

```bash
python demo_test.py
```

4. Deactivate when done:

```bash
deactivate
```

---

## Notes

- Ensure you have Python 3.10+ installed.
- Poetry version 2.1+ is recommended.



This work is done as a Master Thesis at Instituto Superior Técnico.

Project from Miguel Ressurreição