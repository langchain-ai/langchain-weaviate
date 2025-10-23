"""Configure pytest fixtures for Weaviate tests."""

from typing import Any, AsyncGenerator, Generator

import pytest
import pytest_asyncio
import requests
import weaviate


def is_ready(url: str) -> bool:
    """Check if the Weaviate server is ready."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    return False


@pytest.fixture(scope="session")
def weaviate_client(docker_ip: Any, docker_services: Any) -> Generator[Any, None, None]:
    """Create a session-scoped Weaviate client."""
    http_port = docker_services.port_for("weaviate", 8080)
    grpc_port = docker_services.port_for("weaviate", 50051)
    url = f"http://{docker_ip}:{http_port}"

    ready_endpoint = url + "/v1/.well-known/ready"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_ready(ready_endpoint)
    )

    client = weaviate.WeaviateClient(
        connection_params=weaviate.connect.ConnectionParams.from_params(
            http_host=docker_ip,
            http_port=http_port,
            http_secure=False,
            grpc_host=docker_ip,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
    )

    client.connect()
    yield client
    client.close()


@pytest_asyncio.fixture(scope="function")
async def weaviate_client_async(
    docker_ip: Any, docker_services: Any
) -> AsyncGenerator[weaviate.WeaviateAsyncClient, None]:
    """Create a function-scoped async Weaviate client."""
    http_port = docker_services.port_for("weaviate", 8080)
    grpc_port = docker_services.port_for("weaviate", 50051)
    url = f"http://{docker_ip}:{http_port}"

    ready_endpoint = url + "/v1/.well-known/ready"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_ready(ready_endpoint)
    )

    client = weaviate.WeaviateAsyncClient(
        connection_params=weaviate.connect.ConnectionParams.from_params(
            http_host=docker_ip,
            http_port=http_port,
            http_secure=False,
            grpc_host=docker_ip,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
    )

    await client.connect()
    yield client
    await client.close()
