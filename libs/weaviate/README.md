# langchain-weaviate

## About

This package contains the [Weaviate](https://github.com/weaviate/weaviate) integrations for [LangChain](https://github.com/langchain-ai/langchain).

- **Weaviate** is an open source, AI-native vector database that helps developers create intuitive and reliable AI-powered applications.
- **LangChain** is a framework for developing applications powered by language models.

Using this package, LangChain users can conveniently set Weaviate as their vector store to store and retrieve embeddings.

## Requirements

To use this package, you need to have a running Weaviate instance.

Weaviate can be [deployed in many different ways](https://weaviate.io/developers/weaviate/starter-guides/which-weaviate) such as in containerized environments, on Kubernetes, or in the cloud as a managed service, on-premises, or through a cloud provider such as AWS or Google Cloud.

The deployment method to choose depends on your use case and infrastructure requirements.

Two of the most common ways to deploy Weaviate are:
- [Docker Compose](https://weaviate.io/developers/weaviate/installation/docker-compose)
- [Weaviate Cloud Services (WCS)](https://console.weaviate.cloud)

## Installation and Setup

As an integration package, this assumes you have already installed LangChain. If not, please refer to the [LangChain installation guide](https://python.langchain.com/docs/get_started/installation).

Then, install this package:

```bash
pip install langchain-weaviate
```

## Usage

Please see the included [Jupyter notebook](docs/vectorstores.ipynb) for an example of how to use this package.

## Further resources

- [LangChain documentation](https://python.langchain.com/docs)
- [Weaviate documentation](https://weaviate.io/developers/weaviate)
