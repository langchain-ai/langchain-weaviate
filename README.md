# langchain-weaviate

This package contains the LangChain integrations for Weaviate through their `weaviate-client` SDK.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install langchain-weaviate
```

## How to make a release
1. Run `poetry version patch` (or `minor` or `major` as appropriate) to update the version number in `pyproject.toml`.

2. Make a PR with the changes in `pyproject.toml` and merge it.

3. In the `main` branch, run `make tag` to trigger the release workflow.