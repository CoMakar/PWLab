include .build_info

pyi=pipenv run pyinstaller
version=v$(BUILD_VERSION)

wavorizer:
	$(pyi) wavorizer.spec

hexng:
	$(pyi) hexng.spec
	
wavovideo:
	$(pyi) wavovideo.spec

pack:
	if not exist bin mkdir bin
	xcopy .\dist\hexng.exe .\bin\${BUILD_NAME}\hexng\\ /Y
	xcopy .\dist\wavorizer.exe .\bin\${BUILD_NAME}\wavorizer\\ /Y
	xcopy .\dist\wavovideo.exe .\bin\${BUILD_NAME}\wavovideo\\ /Y
	if not exist .versions mkdir .versions
	tar -cvf "./.versions/${BUILD_NAME}_${version}_${BUILD_PLATFORM}.zip" -C "./bin/" "${BUILD_NAME}"

clear:
	if exist build rd /s /q build
	if exist dist rd /s /q dist
	if exist bin rd /s /q bin

all: hexng wavorizer wavovideo pack clear