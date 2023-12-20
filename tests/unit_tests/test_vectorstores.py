from langchain_weaviate.vectorstores import WeaviateVectorStore
import pytest
import requests


def is_ready(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False

@pytest.fixture(scope="function")
def weaviate_host(docker_ip, docker_services):
    port = docker_services.port_for("weaviate", 8080)
    url = "http://{}:{}".format(docker_ip, port)

    ready_endpoint = url + "/v1/.well-known/ready"
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_ready(ready_endpoint)
    )

    return url
