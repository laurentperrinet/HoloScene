default: py

py: SimpleReceptiveField.py
	python SimpleReceptiveField.py

zip: 
	zip SimpleReceptiveField.zip pysight.py SimpleReceptiveField.py Makefile setup.py

edit: 
	mvim pysight.py SimpleReceptiveField.py Makefile setup.py

app: SimpleReceptiveField.py
# 	python setup.py sdist --format=zip
# 	python setup.py egg_info
# 	python setup.py bdist_egg
	python setup.py py2app --iconfile=icon.icns
#	python setup.py py2app --iconfile=icon.icns --no-strip


appzip: app
	cd dist && zip -r SimpleReceptiveField.app.zip SimpleReceptiveField.app

dist: appzip zip


clean: 
	rm -fr dist build  SimpleReceptiveField.zip 

.PHONY: clean 
