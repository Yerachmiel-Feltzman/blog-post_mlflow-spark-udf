### Basic needed setup

- Python 3.10 (if you have Pyenv installed, this project comes bundled with a `.python-version` file)
- Poetry (tested with version 1.4.0)
- Conda (tested with version 23.1.0)

### To reproduce the blog post workflow
1. Install dependencies:

```
poetry install
```

2. Register the model:

```
poetry run python workflow_examples/register_model.py
```

3. Test inference with running environment and see the error:

```
poetry run python workflow_examples/register_model.py
```