# I followed this article's recommendations
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

# The builder image, used to build the virtual environment
FROM python:3.10-buster as builder

RUN pip install poetry==1.8.4

# POETRY_VIRTUALENVS_CREATE=1: Makes sure that environment will be as isolated as possible and above all that
# installation will not mess up with the system Python or, even worse, with Poetry itself.
# POETRY_CACHE_DIR: When removing the cache folder, make sure this is done in the same RUN command. If it’s done in a
# separate RUN command, the cache will still be part of the previous Docker layer (the one containing poetry install )
# effectively rendering your optimization useless.

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /root

COPY pyproject.toml poetry.lock ./

# First, we install only the dependencies. This way, we can cache this layer and avoid re-installing dependencies
# every time we change our application code.
# Because poetry will complain if a README.md is not found, we create a dummy one.
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to run the code
FROM python:3.10-slim-buster as runtime

LABEL image.author.name="Judith Bernett"
LABEL image.author.email="judith.bernett@tum.de"

ENV VIRTUAL_ENV=/root/.venv \
    PATH="/root/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy all relevant code

COPY drevalpy ./drevalpy
COPY create_report.py ./
COPY README.md ./
COPY run_suite.py ./
COPY setup.py ./
COPY pyproject.toml ./
COPY poetry.lock ./

# Install drevalpy
RUN pip install .