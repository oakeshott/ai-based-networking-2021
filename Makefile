
main:
	python main.py

train:
	python estimation.py -i similarity_measures/train --train

test:
	./scripts/test.sh

preprocessing:
	./scripts/preprocessing.sh

similarity:
	./scripts/similarity.sh

download:
	wget --http-user=$(USER) --http-passwd=$(PASSWORD) https://www.ieice.org/cs/rising/jpn/2021/itu-nec/dataset_and_issue.tar.gz
	tar zxvf dataset_and_issue.tar.gz

unzip:
	tar zxvf similarity_measures.tar.gz

