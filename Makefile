.PHONY: install install-dev compile lint test smoke clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

# 静态语法检查：覆盖全部 Python 文件
compile:
	$(PYTHON) -m py_compile $$(find . -name "*.py" -not -path "./.git/*")

# Shell 脚本语法检查
bashcheck:
	@for f in $$(find . -name "*.sh" -not -path "./.git/*"); do bash -n "$$f" || exit 1; done
	@echo "all shell scripts passed bash -n"

lint:
	ruff check src tests

# 单元测试（默认不依赖 GPU，可在 CPU 上运行）
test:
	$(PYTHON) -m unittest discover -s tests -v

# 全链路冒烟测试（CPU 小样本）
smoke:
	bash scripts/smoke_test.sh

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
