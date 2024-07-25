FROM ubuntu:22.04

# Install necessary packages
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    gcc \
    python3-dev \
    python3-setuptools \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pipenv
RUN pip3 install --upgrade pip

# Set the working directory
WORKDIR /root

# Copy Pipfile and Pipfile.lock
COPY ./optimize-nextflow .

RUN pip install --no-cache-dir .

# Download and install tower-cli tool
RUN curl -fSL https://github.com/seqeralabs/tower-cli/releases/download/v0.9.2/tw-linux-x86_64 -o tw \
    && chmod +x tw \
    && mv tw /usr/local/bin


# Export the following environment variables to be used by the container:
# export TOWER_ACCESS_TOKEN=<token>
# export TOWER_API_ENDPOINT=https://tower.sagebionetworks.org/api
# export TOWER_PROJECT_NAME=Sage-Bionetworks/ntap-add5-project
# export WORKFLOW_RUN_ID=3a39HKtlv7C20a

# Run the container:
# docker run --rm \
# -e TOWER_ACCESS_TOKEN=$TOWER_ACCESS_TOKEN \
# -e TOWER_API_ENDPOINT=$TOWER_API_ENDPOINT  \
# -e TOWER_PROJECT_NAME=$TOWER_PROJECT_NAME \
# -e WORKFLOW_RUN_ID=$WORKFLOW_RUN_ID \
# $IMAGE_ID sh -c \
# 'tw --access-token $TOWER_ACCESS_TOKEN --url $TOWER_API_ENDPOINT --output "json" runs view -w $TOWER_PROJECT_NAME -i $WORKFLOW_RUN_ID metrics > metrics.json && python3 optimize-nextflow.py from-json metrics.json'
