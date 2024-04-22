CONTAINER:= document-comparison
PROJECT:= ally
ifeq ($(origin VERSION),undefined)
VERSION:= develop
endif
ifeq ($(origin NAMESPACE),undefined)
NAMESPACE:= dev
endif
LOCAL_REPO:= localhost:32000
REMOTE_REPO:= ydigital.azurecr.io
IMAGE:=$(PROJECT)/$(NAMESPACE)/$(CONTAINER):$(VERSION)
LIMIT_CPU=4
LIMIT_MEMORY=4g

help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Targets:'
	@echo '  build    	build docker --image-- '
	@echo '  run  		run docker --image-- '
setup:
	wget https://github.com/Y-Digital/code-quality/raw/main/.flake8
	wget https://github.com/Y-Digital/code-quality/raw/main/.pre-commit-config.yaml
	wget https://github.com/Y-Digital/code-quality/raw/main/requirements-dev.txt
	pip install -r requirements-dev.txt
	pre-commit install
build:
	DOCKER_BUILDKIT=1 docker build --network=host --ssh default=$$SSH_AUTH_SOCK -t $(IMAGE) .
run: build
	docker run \
		--env-file=./src/local.env \
		--network=host \
		--cpus $(LIMIT_CPU) \
		--memory $(LIMIT_MEMORY) --memory-swap $(LIMIT_MEMORY) \
		$(IMAGE)
run-bash:
	docker run -it --env-file=./src/local.env --entrypoint=/bin/bash $(IMAGE)
push-local:
	docker tag $(IMAGE) ${LOCAL_REPO}/$(IMAGE)
	docker push ${LOCAL_REPO}/$(IMAGE)
push-remote:
	docker tag $(IMAGE) ${REMOTE_REPO}/$(IMAGE)
	docker push ${REMOTE_REPO}/$(IMAGE)

setup:
	wget https://github.com/Y-Digital/code-quality/raw/main/.flake8
	wget https://github.com/Y-Digital/code-quality/raw/main/.pre-commit-config.yaml
	wget https://github.com/Y-Digital/code-quality/raw/main/requirements-dev.txt
	pip install -r requirements-dev.txt
	pre-commit install
	git config blame.ignoreRevsFile .git-blame-ignore-revs
yeet: build push-remote