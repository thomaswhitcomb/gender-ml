CONTAINER_NAME=gender

.PHONY: all
all: shell

.PHONY: shell
shell:
	docker run -it --rm -v $(realpath .):/gender $(CONTAINER_NAME)  /bin/bash

.PHONY: build
build:
	docker build --no-cache -f Dockerfile -t $(CONTAINER_NAME) .
