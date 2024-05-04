pyi=pipenv run pyinstaller
rmrf=rd /s /q
ifex=if exist

wavorizer:
	$(pyi) wavorizer.spec

hexng:
	$(pyi) hexng.spec

all: wavorizer hexng

clear:
	@$(ifex) build $(rmrf) build
	@$(ifex) dist $(rmrf) dist

.PHONY:	wavorizer hexng