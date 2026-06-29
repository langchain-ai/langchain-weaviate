# 🦜️🔗 LangChain Weaviate

This repository contains 1 package with Weaviate integrations with LangChain:

- [langchain-weaviate](https://pypi.org/project/langchain-weaviate/) integrates [Weaviate](https://weaviate.io/).

[Documentation](https://python.langchain.com/docs/integrations/vectorstores/weaviate/)

## Optional Acceleration with `simsimd`

`langchain-weaviate` supports faster vector math through the optional
[`simsimd`](https://pypi.org/project/simsimd/) library.  
If `simsimd` is installed, cosine distance computations use the accelerated path.  
If it is **not installed**, the package automatically falls back to a pure NumPy/SciPy implementation.

This makes `simsimd` *truly optional* and ensures the integration works in all environments
(including those with older glibc versions).

### Install with acceleration
To install with the optional performance extra:

```bash
pip install "langchain-weaviate[simsimd]"