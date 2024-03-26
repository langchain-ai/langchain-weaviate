### Setup

We recommend that you use [development containers](https://containers.dev/) to contribute to this integration.

Please refer tp the files in the `.devcontainer` folder for the specification of the development container for this repository.

#### Installation

To install all dependencies, run the following command:

```shell
poetry install --with test,dev,lint,test_integration
```

Note: This command is run automatically when you start the development container.

### Testing

To run the tests, use the following command:

```shell
make test
```

### Linting

To run the linter, use the following command:

```shell
make format
```

### Creating a Pull Request

1. Fork the repository.
2. Create a new branch from the `main` branch.
3. Make your changes.
4. Run the tests and linter. Make sure the tests pass.
5. Commit your changes.
6. Push your branch to your fork.
7. Create a pull request.
8. In the pull request, describe your changes and tag the issue e.g. (#123) that it fixes.
9. Label your pull request with one of the following labels:
    - `bug`: If the pull request fixes a bug.
    - `documentation`: If the pull request only changes the documentation.
    - `enhancement`: If the pull request adds new functionality.
    - `quality-improvement`: If the pull request is about enhancements to code or workflows for better quality, robustness, performance, etc.
    
   Additionally, if the pull request is a breaking change, add the `breaking` label.
10. Address review comments, if any.