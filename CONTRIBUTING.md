# Community contributions

We welcome community contributions.

In order to facilitate an easier contribution and review process, we provide the following guidelines. Please review them before creating a pull request.

This will improve the chances of your contribution being accepted, by facilitating a higher code quality, and help the maintainers to review the code as well.

### Setup

We recommend that you use [development containers](https://containers.dev/) to contribute to this integration.

Please refer to the files in the `.devcontainer` folder for the specification of the development container for this repository.

If you are new to dev containers, check out a tutorial for your IDE, such as [this one for VS Code](https://code.visualstudio.com/docs/devcontainers/containers).

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

For minor changes, such as fixing typos in documentation or updating workflows and dependencies, feel free to directly submit a pull request (PR). For more substantial contributions like bug fixes or new features, we encourage you to first have a discussion with the project maintainers. This helps us align on the design of the solution and ensures your efforts are on the right track. You can do this by raising a new issue or volunteering to work on an existing one. Please wait to be assigned an issue before starting your work.

1. Fork the repository.
2. Create a new branch from the `main` branch.
3. Make your changes.
4. Run the tests and linter. Make sure the tests pass.
5. Commit your changes.
6. Push your branch to your fork.
7. Create a PR.
8. In the PR, describe your changes and tag the issue e.g. (#123) that it fixes.
9. Label your PR with one of the following labels:
    - `bug`: If the PR fixes a bug.
    - `documentation`: If the PR only changes the documentation.
    - `enhancement`: If the PR adds new functionality.
    - `quality-improvement`: If the PR is about enhancements to code or workflows for better quality, robustness, performance, etc.
    
   Additionally, if the PR is a breaking change, add the `breaking` label.
10. Address review comments, if any.