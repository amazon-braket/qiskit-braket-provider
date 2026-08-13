# Contributing

**We appreciate all kinds of help, so thank you!**

## Contributing to Qiskit-Braket plugin

Specific details for contributing to this project are outlined below.

### Reporting Bugs and Requesting Features

Users are encouraged to use GitHub Issues for reporting issues and requesting features.

### Ask/Answer Questions and Discuss Qiskit-Braket plugin

Users are encouraged to use GitHub Discussions for engaging with researchers, developers, and other users regarding this project and the provided examples.

### Project Code Style

Code in this repository should conform to PEP8 standards. Style/lint checks are run to validate this. Line length must be limited to no more than 100 characters.

### Pull Request Checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. This project uses [Ruff](https://docs.astral.sh/ruff/)
   for linting and formatting.

   You can run
   ```shell script
   tox -e linters_check
   ```
   from the root of the repository clone for lint and format checks.

2. Other tests will run upon PR submission. You can run 
   ```shell script
   tox 
   ```
   to perform a complete test. For debugging, you can utilize tox with additional positional arguments. For instance:
   ```shell script
   tox -e min-versions -- --exitfirst -s
   ```
   runs the minimal environment, but will fail on first failed test will all print outputs to STDOUT. 


### PR Title Format

We use PR titles to update the project version number and generate changelog entries.
The PR title is used as the commit message when merging a PR, so it's important for
PR titles to follow the right format. Valid PR titles include a prefix, separated from
the rest of the message by a colon and a space. Here are a few examples:

```
feature: support new parameter for `xyz`
fix: fix flake8 errors
documentation: add documentation for `xyz`
```

Valid prefixes are listed in the table below.

| Prefix          | Use for...                                                                                     |
|----------------:|:-----------------------------------------------------------------------------------------------|
| `breaking`      | Incompatible API changes.                                                                      |
| `deprecation`   | Deprecating an existing API or feature, or removing something that was previously deprecated.   |
| `feature`       | Adding a new feature.                                                                          |
| `fix`           | Bug fixes.                                                                                     |
| `change`        | Any other code change.                                                                         |
| `documentation` | Documentation changes.                                                                         |
| `infra`         | Infrastructure and CI/CD changes.                                                              |
| `test`          | Test-only changes.                                                                             |

Some of the prefixes allow abbreviation; `break`, `feat`, `depr`, and `doc` are all valid.
If you omit a prefix, the PR title check will fail.
